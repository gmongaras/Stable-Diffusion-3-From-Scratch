# Change huggingface directory
import os
os.environ["HF_HOME"] = "data/cache"
import multiprocessing
from multiprocessing import Process
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.language_model.llava_llama import LlavaConfig
from PIL import Image
import requests
import copy
import pandas as pd
import io
from tqdm import tqdm
import sys
import pickle

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 999999 * (1024**2)




input_dir = "data/Imagenet21K/data/"
output_dir = "data/Imagenet21K/data_recap/"
img_col = "image"
caption_col = "class"
new_caption_col = "recaption"
new_caption_col_short = "recaption_short"
delete_during = True # Do not keep the old parquets without the recaptions
batch_size = 64





if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def recaption_gpu(parquets, gpu_num):
    # Load in the llava model
    pretrained = "lmms-lab/llama3-llava-next-8b"
    model_name = "llava_llama3"
    device = f"cuda:{gpu_num}"
    device_map = device#"auto"
    overwrite_config = {"tokenizer_padding_side": "left"}
    llava_tokenizer, llava_model, llava_image_processor, llava_max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, load_8bit=False, cache_dir="data/cache", overwrite_config=overwrite_config)
    llava_model = llava_model.to(device)
    llava_max_new_tokens = 1024
    llava_tokenizer.pad_token = llava_tokenizer.eos_token
    llava_model.eval()
    llava_model.tie_weights()

    # Load in the llama model
    with open(".env", "r") as f:
        token = f.read().strip()
    from transformers import pipeline
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        model_kwargs={"cache_dir": "data/cache"},
        device_map=f'cuda:{gpu_num}',
        token=token,
    )
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.model_max_length = 2048

    # Used to generate a prompt for the llava model
    llava_prompt_gen = lambda alt_text :f"""
    Please make a detailed but succinct caption of this image. If you see text or objects, be sure to describe them in detail along with any other aspects of the foreground and background. If there is any important text in the image, include it in the caption. As a hint, here is the alt-text attribute of the image, which may or may not have to do with the image:

    Hint:
    \`\`\`
    {alt_text}
    \`\`\`
    """
    # Used to generate a prompt for the llama model
    llama_prompt_gen = lambda img_prompt: f"""
    Please take the following image caption and attempt to distill it into a single sentence. Remove any redundant lines or descriptions and make it a maximum of 40 words in length.

    \`\`\`
    {img_prompt}
    \`\`\`

    Please only write the caption and no other text.
    """

    # Repeated llava openings to be removed
    REPEATED_OPENINGS = [
    ('The image showcases ', ''),
    ('The image portrays ', ''),
    ('The image appears to be ', ''),
    ('The image is ', ''),
    ('The image depicts ', ''),
    ('The image features ', ''),
    ('The image captures ', ''),
    ('The image shows ', ''),
    ('The image displays ', ''),
    ('The image presents ', ''),
    ('This image showcases ', ''),
    ('This image portrays ', ''),
    ('This image appears to be ', ''),
    ('This image is ', ''),
    ('This image depicts ', ''),
    ('This image features ', ''),
    ('This image captures ', ''),
    ('This image shows ', ''),
    ('This image displays ', ''),
    ('This image presents ', ''),
    ('In this picture, ', ''),
    ('In this artwork, ', 'Artwork of '),
    ('In this illustration, ', 'Illustration of '),
    ('In this depiction, ', ''),
    ('In this piece, ', ''),
    ('In this image, ', ''),
    ('In this art piece, ', 'Art of '),
    ('In this scene, ', ''),
    ('In the picture, ', ''),
    ('In the artwork, ', 'Artwork of '),
    ('In the illustration, ', 'Illustration of '),
    ('In the depiction, ', ''),
    ('In the piece, ', ''),
    ('In the image, ', ''),
    ('In the art piece, ', 'Art of '),
    ('In the scene, ', ''),
    ]
    def postprocess_caption(caption: str):
        for often_repeated, replacer in REPEATED_OPENINGS:
            if often_repeated in caption:
                caption = caption.replace(often_repeated, replacer, 1).capitalize()
        return caption.strip()
    # Failed captions have the following repetative phrases
    to_reformats = [' no text', ' other objects', ' additional objects', ' no objects ', 'alt-text']


    # Used to resize images so the largest side is at most 512 pixels and no smaller than 256 pixels
    def resize_image(image):
        # No larger than 512 pixels
        if image.width > image.height:
            if image.width > 512:
                image = image.resize((512, int(image.height / image.width * 512)))
        else:
            if image.height > 512:
                image = image.resize((int(image.width / image.height * 512), 512))

        # # No smaller than 256 pixels
        # if image.width < image.height:
        #     if image.width < 256:
        #       image = image.resize((256, int(image.height / image.width * 256)))
        # else:
        #     if image.height < 256:
        #       image = image.resize((int(image.width / image.height * 256), 256))
        return image


    # Get captions given images and their alt-text
    def get_captions(images, captions):
        llava_model.config.image_aspect_ratio = "pad"
        # images = [resize_image(image) for image in images]
        image_tensor = process_images(images, llava_image_processor, llava_model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "llava_llama_3"
        conv_templates[conv_template].tokenizer = None
        questions = [DEFAULT_IMAGE_TOKEN + llava_prompt_gen(caption) for caption in captions]
        prompts = []
        input_ids = []
        for question in questions:
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.tokenizer = llava_tokenizer
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            prompts.append(prompt)
            input_ids.append(tokenizer_image_token(prompt, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device))
        
        # Pad the sequences after tokenizing (pad on left)
        attention_mask = []
        max_len = max([ids.shape[1] for ids in input_ids])
        padding_id = llava_tokenizer.pad_token_id
        for i in range(len(input_ids)):
            # What is the length of the sequence
            length = input_ids[i].shape[1]

            # How much padding is needed
            padding = max_len - length

            # Add the padding
            input_ids[i] = torch.cat([torch.full((1, padding), padding_id, device=device), input_ids[i]], dim=1)

            # Create mask
            attention_mask.append(torch.ones(max_len, device=device))
            attention_mask[-1][:padding] = 0
        
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attention_mask = torch.stack(attention_mask).to(device)
        image_sizes = [image.size for image in images]

        cont = llava_model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            # temperature=0,
            modalities=["image"]*input_ids.shape[0],
            # max_length=llava_max_length,
            # Max new tokens
            max_new_tokens=llava_max_new_tokens,
            pad_token_id=llava_tokenizer.eos_token_id,
        )


        text_outputs = llava_tokenizer.batch_decode(cont, skip_special_tokens=True)
        text_outputs = [postprocess_caption(caption) for caption in text_outputs]

        # The caption failed if there are a lot of repeated phrases
        for i, caption in enumerate(text_outputs):
            # How many times a phrase is repeated
            repeats = sum([caption.count(reformat) for reformat in to_reformats])
            if repeats > 5:
                text_outputs[i] = None
            
            # Is the sequence repeating too much
            if len(set(caption.split())) < 3:
                text_outputs[i] = None

        return text_outputs



    # Used to make the caption shorter
    @torch.no_grad()
    @torch.inference_mode()
    def postprocess_caption_llama(captions):
        # Create llama prompts
        prompts = [llama_prompt_gen(caption) for caption in captions]

        messages = [[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ] for prompt in prompts]
        outputs = pipe(
            messages,
            eos_token_id=terminators,
            do_sample=False,
            # temperature=0.6,
            # top_p=0.9,
            temperature=None,
            top_p=None,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        outputs = [out[0]["generated_text"][-1]["content"] for out in outputs]
        return outputs



    # Process parquet in batches
    @torch.no_grad()
    @torch.inference_mode()
    def recaption_parquet(parquet):
        # Load in the parquet
        try:
            df = pd.read_parquet(os.path.join(input_dir, parquet))
        except FileNotFoundError:
            return

        # Create new columns
        df[new_caption_col] = None
        df[new_caption_col_short] = None

        # Process rows in batches
        for i in range(0, len(df), batch_size):
            # Get the images and captions
            # images = [Image.open(io.BytesIO(img_bytes)).convert('RGB') for img_bytes in df[img_col].iloc[i:i+batch_size]]
            images = []
            for img_bytes in df[img_col].iloc[i:i+batch_size]:
                try:
                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    images.append(image)
                except ValueError as e:
                    if "Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK" in str(e):
                        print(f"Skipping large image: {e}")
                    else:
                        raise  # Re-raise if it's another ValueError
            captions = df[caption_col].iloc[i:i+batch_size].tolist()

            # Get the new captions
            captions = get_captions(
                images,
                captions,
            )

            # Get the short captions
            recaptions = postprocess_caption_llama(captions)

            # Add the new captions to the dataframe
            df[new_caption_col].iloc[i:i+batch_size] = captions
            df[new_caption_col_short].iloc[i:i+batch_size] = recaptions

        # Delete the failed captions. That is, the ones with a None caption
        num_failed = df[new_caption_col].isnull().sum()
        print(f"Failed: {num_failed}/{len(df)}")
        df = df.dropna(subset=[new_caption_col])

        if len(df) < 10:
            assert False, f"No captions were generated for {parquet}"

        # Save the new dataframe
        df.to_parquet(os.path.join(output_dir, parquet))

        # Delete the old parquet
        if delete_during:
            os.remove(os.path.join(input_dir, parquet))

        del df




    for parquet in tqdm(parquets):
        recaption_parquet(parquet)




# if __name__ == "__main__":
#     # Get all parquets in the input directory
#     parquets = [file for file in os.listdir(input_dir) if file.endswith(".parquet")]

#     # Split evenly across GPUs
#     parquets_split = [parquets[i::num_gpus] for i in range(num_gpus)]

#     # Start the processes
#     processes = []
#     multiprocessing.set_start_method("spawn")
#     for gpu_num in reversed(range(num_gpus)):
#         p = Process(
#             target=recaption_gpu,
#             args=(parquets_split[gpu_num], gpu_num)
#         )
#         p.start()
#         processes.append(p)
#         time.sleep(50)

#     # Wait for all processes to finish
#     while True:
#         if all([not p.is_alive() for p in processes]):
#             break
#         time.sleep(1)
#     # for p in processes:
#     #     p.join()



from multiprocessing import Process
import os

def process_on_gpu(parquets, gpu_num):
    """
    Function to process parquets on a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    recaption_gpu(parquets, gpu_num)

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")

    # # Get all parquets in the input directory
    # parquets = [file for file in os.listdir(input_dir) if file.endswith(".parquet")]

    # # Split parquets evenly across GPUs
    # parquets_split = [parquets[i::num_gpus] for i in range(num_gpus)]
    # # Write list to pickle file
    # with open("parquets_split.pkl", "wb") as f:
    #     pickle.dump(parquets_split, f)

    # # Create processes for each GPU
    # processes = []
    # for gpu_num, parquets_for_gpu in enumerate(parquets_split):
    #     if parquets_for_gpu:  # Ensure there are files for this GPU
    #         p = Process(target=process_on_gpu, args=(parquets_for_gpu, gpu_num))
    #         processes.append(p)
    #         p.start()

    # # Wait for all processes to finish
    # for p in processes:
    #     p.join()


    # Get batch_num and gpu_num command line argument
    batch_num = int(sys.argv[1])
    gpu_num = int(sys.argv[2])

    # Load in the parquets split pickle file
    with open(f"data/Imagenet21K/parquets_split.pkl", "rb") as f:
        parquets_split = pickle.load(f)

    # Process the parquets
    process_on_gpu(parquets_split[batch_num], gpu_num)
