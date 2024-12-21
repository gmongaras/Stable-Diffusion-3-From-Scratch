from transformers import CLIPProcessor, CLIPModel
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








class Data:
    images = None
    text = None

    def __init__(self, images, text, dtype=torch.float16, device=torch.device("cpu")):
        self.images = images.to(dtype=dtype, device=device)
        self.text = text.to(dtype=dtype, device=device)

    def to(self, dtype=torch.float16, device=torch.device("cpu")):
        self.images = self.images.to(dtype=dtype, device=device)
        self.text = self.text.to(dtype=dtype, device=device)
        return self
    









def wait_gpu_n(n, device, data_queue):
    # Wait for a request flag from GPU
    request_flag = torch.tensor([0], device=device)
    dist.irecv(request_flag, src=n).wait()

    if request_flag.item() == 1:  # If GPU requested data
        print(f"Send process: Received request signal from GPU {n}.")
        # while data_queue.empty():
        #     time.sleep(0.01)
        # if not data_queue.empty():
        # Get data from the queue
        next_data = data_queue.get()
        # Send data to GPU
        dist.send(next_data.images, dst=n)
        dist.send(next_data.text, dst=n)
        print(f"Send process: Sent data to GPU {n}.")
        # else:
        #     print("Send process: No data in queue to send.")



# This function will run forever and continually send data to the other GPUs
@torch.no_grad()
@torch.inference_mode()
def send_data_process(data_queue, device, gpu_num):
    """Separate process to handle data transfer."""
    dist.init_process_group(backend="nccl", init_method="env://", world_size=3, rank=0)
    torch.cuda.set_device(device)
    while True:
        # Wait for GPU
        wait_gpu_n(gpu_num, device, data_queue)
        












# Singleton class for VAE + CLIP + T5
class VAE_T5_CLIP:
    _instance = None  # Class variable to store the single instance
    _initialized = False  # Class variable to store the initialization status

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         # Only create the instance on rank 0
    #         if dist.get_rank() == 0:
    #             cls._instance = super(VAE_T5_CLIP, cls).__new__(cls)
    #         # Broadcast the instance reference to all other ranks
    #         dist.barrier()
    #         dist.broadcast_object_list([cls._instance], src=0)
    #     return cls._instance
    

    def __init__(self, batch_size, offload_device, max_in_buffer=30, num_batches=2):
        if not self._initialized:
            self._initialized = True

            # Offloading all models to a single device
            self.device = offload_device
            self.batchSize = batch_size
            self.max_in_buffer = max_in_buffer
            self.num_batches = num_batches


            # Load in the VAE
            self.VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir="./pretrained_models/VAE", device=self.device).eval()
            self.VAE_downsample = 8
            # Freeze all VAE parameters
            for param in self.VAE.parameters():
                param.requires_grad = False
            # Store locally to prevent issues with DDP
            self.VAE = torch.compile(self.VAE.eval()).to(dtype=torch.float16, device=self.device)




            # Load class to string dictionary
            with open('data/imagenet_class_to_string.pkl', 'rb') as f:
                class_to_string = pickle.load(f)
                self.class_to_string = {}
                for k, v in class_to_string.items():
                    self.class_to_string[int(k)] = v



            # CLIP L/4 - https://huggingface.co/openai/clip-vit-large-patch14
            CLIPL4 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./models/CLIP")
            self.CLIPL4 = CLIPL4.text_model
            self.CLIPL4_proj = CLIPL4.text_projection
            del CLIPL4
            self.CLIPL4_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./models/CLIP")
            for param in self.CLIPL4.parameters():
                param.requires_grad = False
            for param in self.CLIPL4_proj.parameters():
                param.requires_grad = False
            self.CLIPL4 = self.CLIPL4.eval().half().to(self.device)
            self.CLIPL4_proj = self.CLIPL4_proj.eval().half().to(self.device)

            # # 
            # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', precision="fp16", device="cpu")
            # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

            # Load data forever
            self.load_data()



    @torch.no_grad()
    @torch.inference_mode()
    def forward_VAE_and_sample(self, x):
        # 1. Encode
        # 2. Sample from the latent distribution
        # 3. Normalize the latent representation
        return self.VAE.encode(x).latent_dist.sample() * self.VAE.config.scaling_factor
    











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
        dataset_ = torchvision.datasets.ImageNet
        pth = "./data/ImageNet12"
        try:
            dataset = dataset_(pth, split="train", transform=transforms)
        except:
            dataset = dataset_(pth, split="train", transform=transforms, download=True)
        def collate_fn(batch):
            return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch])
        data_loader = DataLoader(dataset, batch_size=self.batchSize*self.num_batches,
            pin_memory=True,
            drop_last=False, 
            sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=9999999999999),

            num_workers=10,
            prefetch_factor=10,
            persistent_workers=True,
            collate_fn=collate_fn
        )

        ctx = get_context("spawn")

        # Use multiprocessing Queue to safely share data
        data_queue = ctx.Queue(maxsize=self.max_in_buffer)

        # Start the send_data process for each GPU
        self.send_data_proc_1 = ctx.Process(target=send_data_process, args=(data_queue, self.device, 1))
        self.send_data_proc_1.start()
        self.send_data_proc_2 = ctx.Process(target=send_data_process, args=(data_queue, self.device, 2))
        self.send_data_proc_2.start()

        # # Have a thread continually send data to the other GPUs
        # self.thread = threading.Thread(target=self.send_data)
        # self.thread.start()

        # Iterate forever
        for data in data_loader:
            print(data_queue.qsize())

            # Wait until there is space in the queue
            while data_queue.full():
                time.sleep(0.01)  # Avoid busy-waiting

            batch_x_0, batch_class = data
            batch_x_0 = batch_x_0.to(dtype=torch.float16, device=self.device)
            batch_class = batch_class.to(dtype=torch.float16, device=self.device)

            # # Randomly resize batch between 192 and 256 ( just to have a little variance)
            # # size = np.random.choice([i for i in range(192, 256+1, 16)])
            # size = 256
            # batch_x_0 = torch.nn.functional.interpolate(batch_x_0, size=(size, size), mode="bilinear")

            # Encode batch using VAE - downsample by a factor of 8
            # Get sample from latent distribution using the reparameterization trick
            batch_x_0 = self.forward_VAE_and_sample(batch_x_0)

            # Map each class to a string
            batch_class_enc = [self.class_to_string[int(c)] for c in batch_class]
            # Tokenize the class strings
            batch_class_enc = self.CLIPL4_processor(batch_class, return_tensors="pt", padding=True, truncation=True).to(device=self.device)
            # Encode text using CLIP L/4
            CLIPL4_output = self.CLIPL4(**batch_class_enc)
            CLIPL4_hidden = CLIPL4_output.last_hidden_state
            CLIPL4_pooled = CLIPL4_output.pooler_output

            # # Decode the sample
            # if self.dev == "cpu":
            #     batch_x_0_ = self.model.VAE.decode(batch_x_0).sample.clamp(-1, 1)
            # else:
            #     batch_x_0_ = self.model.module.VAE.decode(batch_x_0).sample.clamp(-1, 1)

            # # Save image
            # torchvision.utils.save_image((batch_x_0_[0]+1)/2, f"sample1.png")
            # torchvision.utils.save_image((batch_x_0_[1]+1)/2, f"sample2.png")
            # torchvision.utils.save_image((batch_x_0_[2]+1)/2, f"sample3.png")

            # Add to the buffer
            batch_x_0 = batch_x_0.split(self.batchSize)
            batch_class = batch_class.split(self.batchSize)
            for i in range(len(batch_x_0)):
                data_queue.put(Data(images=batch_x_0[i], text=batch_class[i], dtype=torch.float16, device=self.device))
            # data_queue.put(Data(images=batch_x_0, text=batch_class, dtype=torch.float16, device=self.device))
            # print("Main: Added data to queue.")






    # This function will return a batch of data and remove it from the buffer
    @torch.no_grad()
    @torch.inference_mode()
    def get_data(self):
        while len(self.data_buffer) == 0:
            pass
        return self.data_buffer.pop(0)
