import torchvision.transforms.functional
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, BitsAndBytesConfig, ModernBertModel
from transformers.models.gemma2.modeling_gemma2 import Gemma2Model
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
import torch_tensorrt
import open_clip
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
# from src.Autoencoder.autoencoder_kl import AutoencoderKL
import threading
import torchvision
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np
import time
import torch.multiprocessing as mp
from torch.multiprocessing import get_context
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
import io
import PIL
from PIL import Image
from PIL import PngImagePlugin
import os
import random
import src.helpers.dataset_utils as dataset_utils
os.environ["TORCH_DYNAMO_MULTI_GPU_SAFE"] = "1"

# Needed to prevent error with large text chunks - I just set it to a shit ton
PngImagePlugin.MAX_TEXT_CHUNK = 1000000 * 1024 * 1024









class Data:
    images = None
    text = None
    text_pooled = None

    def __init__(self, images, text, text_pooled, dtype=torch.bfloat16, device=torch.device("cpu")):
        self.images = images.to(dtype=dtype, device=device)
        self.text = text.to(dtype=dtype, device=device)
        self.text_pooled = text_pooled.to(dtype=dtype, device=device)

    def to(self, dtype=torch.bfloat16, device=torch.device("cpu")):
        self.images = self.images.to(dtype=dtype, device=device)
        self.text = self.text.to(dtype=dtype, device=device)
        self.text_pooled = self.text_pooled.to(dtype=dtype, device=device)
        return self
    




def resize_nearest_multiple(x, v):
    """Resize image so that each side is the nearest multiple of v."""
    c, h, w = x.shape
    new_h = ((h + v - 1) // v) * v  # Round up to nearest multiple of v
    new_w = ((w + v - 1) // v) * v
    return torchvision.transforms.functional.resize(x, (new_h, new_w))






def wait_gpu_n(n, device, parent_conn):
    # Get data from parent data loader
    next_data = parent_conn.recv()

    # Send data to model GPU
    dist.send(next_data["images"], dst=n)
    dist.send(next_data["text"], dst=n)
    dist.send(next_data["text_pooled"], dst=n)

    return

    # Wait for a request flag from GPU
    request_flag = torch.tensor([0], dtype=torch.bool, device=device)
    dist.irecv(request_flag, src=n).wait()

    if request_flag.item() == 1:  # If GPU requested data
        # print(f"Send process: Received request signal from GPU {n}.")
        # while data_queue.empty():
        #     time.sleep(0.01)
        # if not data_queue.empty():
        # Get data from the queue
        # next_data = data_queue.get()

        # # If the model requests data faster than the dataloader is preparing it,
        # # this can run into a race condition.
        # total_seconds = 0
        # while not parent_conn.poll():
        #     time.sleep(0.1)  # Small delay to prevent busy waiting
        #     total_seconds += 0.1
        #     if total_seconds == 60: # Should never take a whole minute
        #         raise(f"Issue on gpu {n} when waiting for data from parent")
        # next_data = parent_conn.recv()

        # Send data to GPU
        dist.send(next_data["images"], dst=n)
        dist.send(next_data["text"], dst=n)
        dist.send(next_data["text_pooled"], dst=n)
        # print(f"Send process: Sent data to GPU {n}.")
        # else:
        #     print("Send process: No data in queue to send.")



# This function will run forever and continually send data to the other GPUs
@torch.no_grad()
@torch.inference_mode()
def send_data_process(parent_conn, device, rank, world_size, gpu_num):
    """Separate process to handle data transfer."""
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(device)
    while True:
        # Wait for GPU
        wait_gpu_n(gpu_num, device, parent_conn)
        





REPEATED_OPENINGS = [
    ('the image showcases ', ''),
    ('the image portrays ', ''),
    ('the image appears to be ', ''),
    ('the image is ', ''),
    ('the image depicts ', ''),
    ('the image features ', ''),
    ('the image captures ', ''),
    ('the image shows ', ''),
    ('the image displays ', ''),
    ('the image presents ', ''),
    ('this image showcases ', ''),
    ('this image portrays ', ''),
    ('this image appears to be ', ''),
    ('this image is ', ''),
    ('this image depicts ', ''),
    ('this image features ', ''),
    ('this image captures ', ''),
    ('this image shows ', ''),
    ('this image displays ', ''),
    ('this image presents ', ''),
    ('in this picture, ', ''),
    ('in this artwork, ', 'artwork of '),
    ('in this illustration, ', 'illustration of '),
    ('in this depiction, ', ''),
    ('in this piece, ', ''),
    ('in this image, ', ''),
    ('in this art piece, ', 'art of '),
    ('in this scene, ', ''),
    ('in the picture, ', ''),
    ('in the artwork, ', 'artwork of '),
    ('in the illustration, ', 'illustration of '),
    ('in the depiction, ', ''),
    ('in the piece, ', ''),
    ('in the image, ', ''),
    ('in the art piece, ', 'art of '),
    ('in the scene, ', ''),
    ]






# Class for VAE + CLIP + T5
class VAE_T5_CLIP:
    def __init__(self, bucket_indices_path, data_parquet_folder, max_res, batch_size, offload_device, rank, world_size, loader_to_model_gpu, max_in_buffer=30, num_batches=2):
        # Offloading all models to a single device
        self.device = offload_device
        self.loader_to_model_gpu = loader_to_model_gpu
        self.batchSize = batch_size
        self.max_in_buffer = max_in_buffer
        self.num_batches = num_batches
        self.max_res = max_res

        # Get the rank of the current process
        self.rank = rank
        self.world_size = world_size

        # Get the GPUs corresponding to the current process
        self.gpus = loader_to_model_gpu[self.rank]


        # Load in the VAE
        self.VAE = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            subfolder="vae", 
            cache_dir="./models/VAE", 
            device=self.device,
            # attn_implementation="flash_attention_2",
        ).eval()
        # self.VAE = AutoencoderKL.from_single_file(url, config="./models/VAE/FLUX_config.json", cache_dir="./pretrained_models/VAE", device=self.device).eval()
        self.VAE_downsample = 8
        # We don't need the decoder
        del self.VAE.decoder
        self.VAE.decoder = None
        # Freeze all VAE parameters
        for param in self.VAE.parameters():
            param.requires_grad = False
        # Store locally to prevent issues with DDP
        self.VAE = self.VAE.eval().to(dtype=torch.bfloat16, device=self.device)
        # self.VAE = torch.compile(self.VAE, mode="reduce-overhead")
        # Passes image data through the VAE and then samples from the latent distribution
        # @torch.no_grad()
        # @torch.inference_mode()
        def forward_VAE_and_sample(x):
            # 1. Encode
            # 2. Sample from the latent distribution
            # 3. Normalize the latent representation
            return self.VAE.encode(x).latent_dist.sample() * self.VAE.config.scaling_factor + self.VAE.config.shift_factor
        # forward_VAE_and_sample = torch.compile(forward_VAE_and_sample, backend="inductor")
        self.forward_VAE_and_sample = forward_VAE_and_sample




        # Load CLIP model (400M version) - https://huggingface.co/facebook/metaclip-l14-400m
        # Or larger version (2.5B) - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b
        self.CLIP_processor = CLIPProcessor.from_pretrained("facebook/metaclip-l14-400m", cache_dir="./models/CLIP")
        self.CLIP_model = CLIPModel.from_pretrained(
            "facebook/metaclip-l14-400m", 
            cache_dir="./models/CLIP",
            # attn_implementation="flash_attention_2",
        ).to(self.device)
        for param in self.CLIP_model.parameters():
            param.requires_grad = False
        self.CLIP_model = self.CLIP_model.eval().half()
        # We want to use the text projection layer as the final output which
        # also decreases the variance of the output.
        self.CLIP_text_proj = self.CLIP_model.text_projection
        # Delete vision model
        # del self.CLIP_model.vision_model
        # del self.CLIP_model.visual_projection
        # model_CLIP = torch.compile(model_CLIP, backend="inductor")
        # @torch.no_grad()
        # @torch.inference_mode()
        def CLIP_encode_text(text):
            text = self.CLIP_processor(text, return_tensors="pt", padding=True, truncation=True).to(device=self.device)
            return self.CLIP_text_proj(self.CLIP_model.text_model(**text).pooler_output)
        self.CLIP_encode_text = CLIP_encode_text





        # Gemma 2B - https://huggingface.co/google/gemma-2-2b
        try:
            with open(".env", "r") as f:
                token = f.read().strip()
        except:
            with open("../.env", "r") as f:
                token = f.read().strip()
        self.Gemma_tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-2-2b", cache_dir="./models/Gemma2b", legacy=False, token=token, padding_side="right")
        self.Gemma_model = Gemma2Model.from_pretrained(
            "google/gemma-2-2b", 
            torch_dtype=torch.bfloat16,
            cache_dir="./models/Gemma2b", 
            token=token,
            device_map=self.device,
            # attn_implementation="flash_attention_2",
        ).eval().to(self.device)
        # @torch.no_grad()
        # @torch.inference_mode()
        def Gemma_encode_text(text): # Output of shape (B, 77, 2304)
            tokenized = self.Gemma_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(self.device)
            return self.Gemma_model(**tokenized, use_cache=False).last_hidden_state * tokenized["attention_mask"][:, :, None]
        self.Gemma_encode_text = Gemma_encode_text




        # # ByT5 for byte-level text generation - https://huggingface.co/google/byt5-large
        # self.ByT5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large", cache_dir="./models/ByT5", use_fast=True)
        # self.ByT5_model = T5EncoderModel.from_pretrained(
        #     "google/byt5-large",
        #     torch_dtype=torch.bfloat16,
        #     cache_dir="./models/ByT5",
        #     device_map=self.device,
        # ).to_bettertransformer().eval().to(self.device)
        # def T5_generate_text(text):
        #     text = self.ByT5_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(self.device)
        #     return self.ByT5_model(**text).last_hidden_state
        # self.T5_generate_text = T5_generate_text



        # ModernBert as a second model - https://huggingface.co/answerdotai/ModernBERT-large
        self.ModernBert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large", cache_dir="./models/ModernBERT", use_fast=True, padding_side="right")
        self.ModernBert_model = ModernBertModel.from_pretrained(
            "answerdotai/ModernBERT-large",
            torch_dtype=torch.bfloat16,
            cache_dir="./models/ModernBERT",
            device_map=self.device,
        ).eval().to(self.device)
        def ModernBert_generate_text(text):
            text = self.ModernBert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(self.device)
            return self.ModernBert_model(**text).last_hidden_state * text["attention_mask"][:, :, None]
        self.ModernBert_generate_text = ModernBert_generate_text



        torch.cuda.empty_cache()

        # Load data forever
        self.load_data(bucket_indices_path, data_parquet_folder)


        # This should never be reached
        raise "Load data stopped for some reason. This is usually an issue with the shapes on the receiving end (the model) not being the same as the data being sent here (Ex: different batch size)"
    












    # This function will run forever and continually add data to the data buffer
    @torch.no_grad()
    @torch.inference_mode()
    def load_data(self, bucket_indices_path, data_parquet_folder):
        # Create a sampler and loader over the dataset
        transforms = torchvision.transforms.Compose([
            # # Resize so the largest side is 256
            # torchvision.transforms.Resize(256-self.VAE_downsample*2, max_size=256),
            # # Resize the shorter side to 256 with bicubic interpolation
            # torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # # Center crop to 256 x 256
            # torchvision.transforms.CenterCrop((256, 256)),
            # Convert to tensor
            torchvision.transforms.ToTensor(),
            # # Resize to the nearest multiple of the VAE downsample factor * the patch size (2)
            # torchvision.transforms.Lambda(lambda x: resize_nearest_multiple(x, self.VAE_downsample*2)),
            # Data already in range [0, 1]. Make between -1 and 1
            torchvision.transforms.Lambda(lambda x: 2*x - 1.0)
        ])
        # dataset_ = torchvision.datasets.ImageNet
        # pth = "./data/ImageNet12"
        # try:
        #     dataset = dataset_(pth, split="train", transform=transforms)
        # except:
        #     dataset = dataset_(pth, split="train", transform=transforms, download=True)
        # def collate_fn(batch):
        #     return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch])
        # dataset = datasets.load_dataset("parquet", data_files=f"data/Stable_Diffusion_3_Recaption/data/*.parquet", cache_dir="data/cache", split="train")
        ### NOTE: If the above runs into a weird index error, I added to the file "/python_pacakages/datasets/packaged_modules/parquet/parquet.py"
        ###       after line 102 the following lines:
        ###       if "__index_level_0__" in pa_table.column_names:
        ###           pa_table = pa_table.drop(['__index_level_0__'])
        ### Which will remove the index column from the table if it exists
        # dataset = datasets.load_dataset("parquet", data_files=f"data/cc12m_and_imagenet21K/*.parquet", cache_dir="data/cache", split="train")
        dataset = datasets.load_dataset("parquet", data_files=f"{data_parquet_folder}/*.parquet", cache_dir="data/cache", split="train", num_proc=64)
        def transform_img(img):
            img = Image.open(io.BytesIO(img))
            img_ = transforms(img)
            img.close()
            return img_
        def clean_text(text):
            try:
                # Repalce A...
                if random.random() < 0.5:
                    text = text.replace("A ", "").replace("An ", "")
                # Remove repeated openings
                for opening in REPEATED_OPENINGS:
                    text = text.replace(opening[0], opening[1])
                # Remove punctuation at the end
                if text[-1] in [".", ",", "!", "?"] and random.random() < 0.5:
                    text = text[:-1].strip()
                return text
            except:
                return ""
        def collate_fn(batch):
            # 50% chance for normal caption and 50% chance for short caption
            cap_type = "recaption" if random.random() < 0.5 else "recaption_short"
            return torch.stack([transform_img(b["image"]) for b in batch]), \
                [b[cap_type].strip() for b in batch]
            batch = [transform_img(b["image"]) for b in batch], \
                [b["recaption_short"].strip() for b in batch]
            # Get the largest image height and largest image width in the batch
            max_h = max([b.shape[1] for b in batch[0]])
            max_w = max([b.shape[2] for b in batch[0]])
            # Pad all images to the largest image height and largest image width in the batch and get a padding mask
            padded = torch.zeros((len(batch[0]), 3, max_h, max_w), dtype=torch.bfloat16)
            padding_mask = torch.zeros((len(batch[0]), max_h, max_w), dtype=torch.bool)
            for i, b in enumerate(batch[0]):
                padded[i, :, :b.shape[1], :b.shape[2]] = b
                padding_mask[i, :b.shape[1], :b.shape[2]] = True
            padding_mask_downsampled = padding_mask[:, ::self.VAE_downsample, ::self.VAE_downsample]
            return padded, batch[0], batch[1], padding_mask, padding_mask_downsampled
        # data_loader = DataLoader(dataset, batch_size=self.batchSize*self.num_batches,
        #     pin_memory=True,
        #     drop_last=False, 
        #     sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=9999999999999999),

        #     num_workers=10,
        #     prefetch_factor=10,
        #     persistent_workers=True,
        #     collate_fn=collate_fn
        # )
        
        dataset_utils.load_indices(bucket_indices_path, dataset)
        hf_dataset = dataset_utils.HuggingFaceDataset(dataset)
        sampler = dataset_utils.RandomBucketSampler(bucket_indices_path, dataset, self.batchSize*self.num_batches)
        data_loader = DataLoader(hf_dataset, 
            batch_sampler=sampler, 
            pin_memory=True,
            drop_last=False,

            num_workers=10,
            prefetch_factor=10,
            persistent_workers=True,
            collate_fn=collate_fn
        )
        """
        ctx = get_context("spawn")

        # Use multiprocessing Queue to safely share data
        data_queue = ctx.Queue(maxsize=self.max_in_buffer)

        # Start the send_data process for each GPU
        for gpu in self.gpus:
            ctx.Process(target=send_data_process, args=(data_queue, self.device, self.rank, self.world_size, gpu)).start()
        """
        ctx = mp.get_context("spawn")
        processes = []
        pipes = []

        # Start the send_data process for each GPU and keep a connection to each process
        for gpu in self.gpus:
            parent_conn, child_conn = ctx.Pipe(duplex=True)  # Create a Pipe
            pipes.append(parent_conn)
            process = ctx.Process(target=send_data_process, args=(child_conn, self.device, self.rank, self.world_size, gpu))
            processes.append(process)
            process.start()

        # Iterate forever
        for data in data_loader:
            batch_x_0, batch_text = data
            # batch_x_0_, batch_x_0, batch_text, padding_mask, padding_mask_downsampled = data
            batch_x_0 = batch_x_0.to(dtype=torch.bfloat16, device=self.device)
            # batch_x_0_ = batch_x_0_.to(dtype=torch.bfloat16, device=self.device)

            # Encode text using Gemma - (B, 77, 2304)
            text_hidden_Gemma = self.Gemma_encode_text(batch_text)

            # Encode text using ModernBert - (B, 77, 1024)
            text_hidden_ModernBert = self.ModernBert_generate_text(batch_text)
            # Cancat zero padding to make the two embeddings the same size - (B, 77, 2304)
            text_hidden_ModernBert = torch.cat([text_hidden_ModernBert, torch.zeros((text_hidden_ModernBert.shape[0], 77, text_hidden_Gemma.shape[-1]-text_hidden_ModernBert.shape[-1]), dtype=text_hidden_ModernBert.dtype, device=text_hidden_ModernBert.device)], dim=-1)

            # Concatenate the two embeddings along the sequence dimension - (B, 77*2, 2304)
            text_hidden = torch.cat([text_hidden_Gemma, text_hidden_ModernBert], dim=1)

            # Encode batch using VAE - downsample by a factor of 8
            # Get sample from latent distribution using the reparameterization trick
            # Normalize the latent representation
            # (B, 3, L, W) -> (B, 16, L//8, W//8)
            # batch_x_0__out = self.forward_VAE_and_sample(batch_x_0_, (~padding_mask[:, None, :, :]).to(batch_x_0_.device))
            batch_x_0 = self.forward_VAE_and_sample(batch_x_0)
            # Pad latents to be of shape (256//8, 256//8). This is the max size so that
            # the GPU sync works properly.
            batch_x_0 = F.pad(batch_x_0, (0, self.max_res//8-batch_x_0.shape[-1], 0, self.max_res//8-batch_x_0.shape[-2]), value=torch.inf)

            # Get pooled embedding from CLIP - (B, 768)
            text_pooled = self.CLIP_encode_text(batch_text)


            

            # Add to the buffer
            batch_x_0 = batch_x_0.split(self.batchSize)
            text = text_hidden.split(self.batchSize)
            text_pooled = text_pooled.split(self.batchSize)
            # Send data directly to each process
            # for i, n in enumerate(self.gpus):
            #     request_flag = torch.tensor([0], dtype=torch.bool, device=self.device)
            #     dist.irecv(request_flag, src=n).wait()

            #     if request_flag.item() == 1:  # If GPU requested data
            #         # print(f"Send process: Received request signal from GPU {n}.")
            #         # while data_queue.empty():
            #         #     time.sleep(0.01)
            #         # if not data_queue.empty():
            #         # Get data from the queue
            #         # next_data = data_queue.get()

            #         # # If the model requests data faster than the dataloader is preparing it,
            #         # # this can run into a race condition.
            #         # total_seconds = 0
            #         # while not parent_conn.poll():
            #         #     time.sleep(0.1)  # Small delay to prevent busy waiting
            #         #     total_seconds += 0.1
            #         #     if total_seconds == 60: # Should never take a whole minute
            #         #         raise(f"Issue on gpu {n} when waiting for data from parent")
            #         # next_data = parent_conn.recv()

            #         # Send data to GPU
            #         dist.send(batch_x_0[i], dst=n)
            #         dist.send(text[i], dst=n)
            #         dist.send(text_pooled[i], dst=n)
            for i, pipe in enumerate(pipes):
                pipe.send({"images": batch_x_0[i].to(torch.bfloat16), "text": text[i].to(torch.bfloat16), "text_pooled": text_pooled[i].to(torch.bfloat16)})


