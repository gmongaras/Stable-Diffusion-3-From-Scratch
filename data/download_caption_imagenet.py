import torch
import torchvision
from torch.utils.data import DataLoader
import os
import pickle
import random
from tqdm import tqdm
import pandas as pd
import io
from PIL import Image




# Create a sampler and loader over the dataset
transforms = torchvision.transforms.Compose([
    # Convert to bytes
])
dataset_ = torchvision.datasets.ImageNet
pth = "./data/ImageNet12"
out_pth = "./data/final_dataset"
num_per_parquet = 1024
try:
    dataset = dataset_(pth, split="train", transform=transforms)
except:
    dataset = dataset_(pth, split="train", transform=transforms, download=True)
try:
    dataset_val = dataset_(pth, split="val", transform=transforms)
except:
    dataset_val = dataset_(pth, split="val", transform=transforms, download=True)
# Function to convert the Image to bytes
def image_to_bytes(image):
    rbuffer = io.BytesIO()
    image.save(rbuffer, format="PNG")
    return rbuffer.getvalue()
def collate_fn(batch):
    return {
        "id": [999999999] * len(batch),
        "image": [image_to_bytes(b[0]) for b in batch],
        "caption": [f"a photo of a {random.choice(class_to_string[b[1]])}" for b in batch]
    }
data_loader = DataLoader(dataset, batch_size=num_per_parquet,
    pin_memory=True,
    drop_last=False,
    shuffle=True,

    num_workers=1,
    prefetch_factor=10,
    persistent_workers=True,
    collate_fn=collate_fn,
)
data_loader_val = DataLoader(dataset_val, batch_size=num_per_parquet,
    pin_memory=True,
    drop_last=False,
    shuffle=True,

    num_workers=1,
    prefetch_factor=10,
    persistent_workers=True,
    collate_fn=collate_fn,
)


# Load class to string dictionary
with open('data/imagenet_class_to_string.pkl', 'rb') as f:
    class_to_string_ = pickle.load(f)
    class_to_string = {}
    for k, v in class_to_string_.items():
        class_to_string[int(k)] = [i.strip().replace("\"", "") for i in v.split(",") if i.strip() != ""]


# Iterate over the entire dataset and save the images and labels
if not os.path.exists(out_pth):
    os.makedirs(out_pth)
data_len = len(data_loader)
for i, batch in tqdm(enumerate(data_loader), total=data_len):
    # Save the images and labels to parquet
    pd.DataFrame(batch).to_parquet(f"{out_pth}/data_{i}.parquet")

# Save the validation set
for i, batch in tqdm(enumerate(data_loader_val), total=len(data_loader_val)):
    i += data_len
    # Save the images and labels to parquet
    pd.DataFrame(batch).to_parquet(f"{out_pth}/val_{i}.parquet")