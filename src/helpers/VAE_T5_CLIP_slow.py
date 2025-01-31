from transformers import CLIPProcessor, CLIPModel, AutoProcessor
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
import torch_tensorrt
import open_clip
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
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
os.environ["TORCH_DYNAMO_MULTI_GPU_SAFE"] = "1"

# Needed to prevent error with large text chunks - I just set it to a shit ton
PngImagePlugin.MAX_TEXT_CHUNK = 1000000 * 1024 * 1024









class Data:
    images = None
    text = None
    text_pooled = None

    def __init__(self, images, text, text_pooled, dtype=torch.float16, device=torch.device("cpu")):
        self.images = images.to(dtype=dtype, device=device)
        self.text = text.to(dtype=dtype, device=device)
        self.text_pooled = text_pooled.to(dtype=dtype, device=device)

    def to(self, dtype=torch.float16, device=torch.device("cpu")):
        self.images = self.images.to(dtype=dtype, device=device)
        self.text = self.text.to(dtype=dtype, device=device)
        self.text_pooled = self.text_pooled.to(dtype=dtype, device=device)
        return self
    









def wait_gpu_n(n, device, data_queue):
    # Wait for a request flag from GPU
    request_flag = torch.tensor([0], dtype=torch.bool, device=device)
    dist.irecv(request_flag, src=n).wait()

    if request_flag.item() == 1:  # If GPU requested data
        # print(f"Send process: Received request signal from GPU {n}.")
        # while data_queue.empty():
        #     time.sleep(0.01)
        # if not data_queue.empty():
        # Get data from the queue
        next_data = data_queue.get()
        # Send data to GPU
        dist.send(next_data.images, dst=n)
        dist.send(next_data.text, dst=n)
        dist.send(next_data.text_pooled, dst=n)
        # print(f"Send process: Sent data to GPU {n}.")
        # else:
        #     print("Send process: No data in queue to send.")



# This function will run forever and continually send data to the other GPUs
@torch.no_grad()
@torch.inference_mode()
def send_data_process(data_queue, device, rank, world_size, gpu_num):
    """Separate process to handle data transfer."""
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(device)
    while True:
        # Wait for GPU
        wait_gpu_n(gpu_num, device, data_queue)










# Class for VAE + CLIP + T5
class VAE_T5_CLIP:
    def __init__(self, batch_size, offload_device, rank, world_size, loader_to_model_gpu, max_in_buffer=30, num_batches=2):
        # Offloading all models to a single device
        self.device = offload_device
        self.loader_to_model_gpu = loader_to_model_gpu
        self.batchSize = batch_size
        self.max_in_buffer = max_in_buffer
        self.num_batches = num_batches

        # Get the rank of the current process
        self.rank = rank
        self.world_size = world_size

        # Get the GPUs corresponding to the current process
        self.gpus = loader_to_model_gpu[self.rank]


        # Load in the VAE
        self.VAE = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="vae", cache_dir="./models/VAE", device=self.device).eval()
        # self.VAE = AutoencoderKL.from_single_file(url, config="./models/VAE/FLUX_config.json", cache_dir="./pretrained_models/VAE", device=self.device).eval()
        self.VAE_downsample = 8
        # We don't need the decoder
        del self.VAE.decoder
        self.VAE.decoder = None
        # Freeze all VAE parameters
        for param in self.VAE.parameters():
            param.requires_grad = False
        # Store locally to prevent issues with DDP
        self.VAE = self.VAE.eval().to(dtype=torch.float16, device=self.device)

        # Passes image data through the VAE and then samples from the latent distribution
        @torch.no_grad()
        @torch.inference_mode()
        def forward_VAE_and_sample(x):
            # 1. Encode
            # 2. Sample from the latent distribution
            # 3. Normalize the latent representation
            return self.VAE.encode(x).latent_dist.sample() * self.VAE.config.scaling_factor + self.VAE.config.shift_factor
        forward_VAE_and_sample = torch.compile(forward_VAE_and_sample, backend="inductor")
        self.forward_VAE_and_sample = forward_VAE_and_sample




        # Load CLIP model (400M version) - https://huggingface.co/facebook/metaclip-l14-400m
        # Or larger version (2.5B) - https://huggingface.co/facebook/metaclip-h14-fullcc2.5b
        self.CLIP_processor = CLIPProcessor.from_pretrained("facebook/metaclip-l14-400m", cache_dir="./models/CLIP", use_fast=True)
        self.CLIP_model = CLIPModel.from_pretrained("facebook/metaclip-l14-400m", cache_dir="./models/CLIP").to(self.device)
        for param in self.CLIP_model.parameters():
            param.requires_grad = False
        self.CLIP_model = self.CLIP_model.eval().half()
        # Delete vision model
        # del self.CLIP_model.vision_model
        # del self.CLIP_model.visual_projection
        @torch.no_grad()
        @torch.inference_mode()
        def model_CLIP(text):
            return self.CLIP_model.text_model(**text).pooler_output
        # model_CLIP = torch.compile(model_CLIP, backend="inductor")
        def CLIP_encode_text(text):
            text = self.CLIP_processor(text, return_tensors="pt", padding=True, truncation=True).to(device=self.device)
            return model_CLIP(text)
        self.CLIP_encode_text = CLIP_encode_text





        # Gemma 2B - https://huggingface.co/google/gemma-2-2b
        with open(".env", "r") as f:
            token = f.read().strip()
        self.Gemma_tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-2-2b", cache_dir="./models/Gemma2b", legacy=False, token=token)
        self.Gemma_model = Gemma2ForCausalLM.from_pretrained("google/gemma-2-2b", cache_dir="./models/Gemma2b", token=token).eval().to(self.device)
        @torch.no_grad()
        @torch.inference_mode()
        def Gemma_encode_text(text): # Output of shape (B, 128, 2304)
            tokenized = self.Gemma_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(self.device)
            return self.Gemma_model(**tokenized, output_hidden_states=True, num_logits_to_keep=1).hidden_states[-1]
        self.Gemma_encode_text = Gemma_encode_text



        torch.cuda.empty_cache()

        # Load data forever
        self.load_data()
    











    # This function will run forever and continually add data to the data buffer
    @torch.no_grad()
    @torch.inference_mode()
    def load_data(self):
        # Create a sampler and loader over the dataset
        transforms = torchvision.transforms.Compose([
            # Resize the shorter side to 256 with bicubic interpolation
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # Center crop to 256 x 256
            torchvision.transforms.CenterCrop((256, 256)),
            # Convert to tensor
            torchvision.transforms.ToTensor(),
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
        dataset = datasets.load_dataset("parquet", data_files=f"data/Stable_Diffusion_3_Recaption/data/*.parquet", cache_dir="data/cache", split="train")
        def transform_img(img):
            img = Image.open(io.BytesIO(img))
            img_ = transforms(img)
            img.close()
            return img_
        def collate_fn(batch):
            return torch.stack([transform_img(b["image"]) for b in batch]), \
                [b["caption"] for b in batch]
        data_loader = DataLoader(dataset, batch_size=self.batchSize*self.num_batches,
            pin_memory=True,
            drop_last=False, 
            sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=9999999999999999),

            num_workers=10,
            prefetch_factor=10,
            persistent_workers=True,
            collate_fn=collate_fn
        )

        ctx = get_context("spawn")

        # Use multiprocessing Queue to safely share data
        data_queue = ctx.Queue(maxsize=self.max_in_buffer)

        # Start the send_data process for each GPU
        for gpu in self.gpus:
            ctx.Process(target=send_data_process, args=(data_queue, self.device, self.rank, self.world_size, gpu)).start()

        # # Have a thread continually send data to the other GPUs
        # self.thread = threading.Thread(target=self.send_data)
        # self.thread.start()

        # Iterate forever
        for data in data_loader:
            # print(data_queue.qsize())

            # Wait until there is space in the queue
            while data_queue.full():
                time.sleep(0.01)  # Avoid busy-waiting

            batch_x_0, batch_class = data
            batch_x_0 = batch_x_0.to(dtype=torch.float16, device=self.device)

            # Encode text using Gemma - (B, 128, 2304)
            text_hidden = self.Gemma_encode_text(batch_class)

            # Encode batch using VAE - downsample by a factor of 8
            # Get sample from latent distribution using the reparameterization trick
            # Normalize the latent representation
            # (B, 3, L, W) -> (B, 16, L//8, W//8)
            batch_x_0 = self.forward_VAE_and_sample(batch_x_0)

            # Get pooled embedding from CLIP - (B, 768)
            text_pooled = self.CLIP_encode_text(batch_class)

            

            # Add to the buffer
            batch_x_0 = batch_x_0.split(self.batchSize)
            text = text_hidden.split(self.batchSize)
            text_pooled = text_pooled.split(self.batchSize)
            for i in range(len(batch_x_0)):
                data_queue.put(Data(images=batch_x_0[i], text=text[i], text_pooled=text_pooled[i], dtype=torch.float16, device=self.device))






    # This function will return a batch of data and remove it from the buffer
    @torch.no_grad()
    @torch.inference_mode()
    def get_data(self):
        while len(self.data_buffer) == 0:
            pass
        return self.data_buffer.pop(0)
