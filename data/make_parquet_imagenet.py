import torch
import torchvision
from torch.utils.data import DataLoader
import os
import pickle
import random
from tqdm import tqdm




# Create a sampler and loader over the dataset
transforms = torchvision.transforms.Compose([
    # Convert to tensor
    torchvision.transforms.ToTensor(),
])
dataset_ = torchvision.datasets.ImageNet
pth = "./data/ImageNet12"
out_pth = "./data/ImageNet12_Caption"
num_per_pickle = 1024
try:
    dataset = dataset_(pth, split="train", transform=transforms)
except:
    dataset = dataset_(pth, split="train", transform=transforms, download=True)
try:
    dataset_val = dataset_(pth, split="val", transform=transforms)
except:
    dataset_val = dataset_(pth, split="val", transform=transforms, download=True)
def collate_fn(batch):
    return {
        "images": [b[0] for b in batch], 
        "labels": [f"a photo of a {random.choice(class_to_string[b[1]])}" for b in batch]
    }
data_loader = DataLoader(dataset, batch_size=num_per_pickle,
    pin_memory=True,
    drop_last=False,
    shuffle=True,

    num_workers=1,
    prefetch_factor=10,
    persistent_workers=True,
    collate_fn=collate_fn
)
data_loader_val = DataLoader(dataset_val, batch_size=num_per_pickle,
    pin_memory=True,
    drop_last=False,
    shuffle=True,

    num_workers=1,
    prefetch_factor=10,
    persistent_workers=True,
    collate_fn=collate_fn
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
    # Save the images and labels
    with open(f"{out_pth}/data_{i}.pkl", "wb") as f:
        pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)



# Same for the validation set
for i, batch in tqdm(enumerate(data_loader_val), total=len(data_loader_val)):
    i += data_len
    # Save the images and labels
    with open(f"{out_pth}/data_{i}.pkl", "wb") as f:
        pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)