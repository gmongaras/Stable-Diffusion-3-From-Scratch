import torch
import torchvision
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from scipy.linalg import sqrtm
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

if not os.path.exists("FID_results/Imagenet"):
    os.makedirs("FID_results/Imagenet")

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1.0)  # Map to range [-1, 1]
])

# Load dataset
dataset_path = "./data/ImageNet12"
batch_size = 64  # Modify based on available memory
init = 0

try:
    dataset = torchvision.datasets.ImageNet(dataset_path, split="train", transform=transform)
except:
    dataset = torchvision.datasets.ImageNet(dataset_path, split="train", transform=transform, download=True)

data_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(42)), 
    pin_memory=True, 
    drop_last=False, 
    num_workers=10, 
    prefetch_factor=10, 
    persistent_workers=True
)

# Load pre-trained InceptionV3 model
inception = inception_v3(pretrained=True, transform_input=False)
inception.eval()
inception = inception.cuda()

# Disable gradients for model inference
for param in inception.parameters():
    param.requires_grad = False

# Function to calculate activations from InceptionV3
def calculate_activations(loader, model):
    activations = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Calculating Activations"):
            images = images.cuda()  # Move to GPU if available
            preds = model(images).detach().cpu().numpy()
            activations.append(preds)
    return np.concatenate(activations, axis=0)

# Calculate statistics for FID
def calculate_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

# Calculate FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

from torch.utils.data import Subset
from collections import defaultdict

# Load cached indices
if os.path.exists("FID_results/Imagenet/class_indices.npy"):
    class_indices = np.load("FID_results/Imagenet/class_indices.npy", allow_pickle=True).item()
    dataset = Subset(dataset, list(range(len(dataset))))
else:
    # Create indices for each class
    class_indices = defaultdict(list)
    for idx, (_, target) in tqdm(enumerate(dataset), desc="Creating Class Indices", total=len(dataset)):
        class_indices[target].append(idx)

    # Save the indices to a file
    np.save("FID_results/Imagenet/class_indices.npy", class_indices)

# Create DataLoaders for each class
per_class_loaders = {
    class_id: DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=10
    )
    for class_id, indices in class_indices.items()
}

fid_results = {}

# Generate activations for each class
for class_name, loader in per_class_loaders.items():
    # Skip up to the initial class
    if class_name < init:
        continue
    print(f"Processing class: {class_name}")
    activations = calculate_activations(loader, inception)
    mu, sigma = calculate_statistics(activations)
    fid_results[class_name] = {"mu": mu, "sigma": sigma}

# Save the results to a file
np.save("FID_results/Imagenet/fid_results.npy", fid_results)
# Save by class
for class_name, stats in fid_results.items():
    np.save(f"FID_results/Imagenet/{class_name}_mu.npy", stats["mu"])
    np.save(f"FID_results/Imagenet/{class_name}_sigma.npy", stats["sigma"])

# Save or display FID results
print("FID Results:")
for class_name, stats in fid_results.items():
    print(f"{class_name}: Mean = {stats['mu']}, Covariance Matrix Shape = {stats['sigma'].shape}")
