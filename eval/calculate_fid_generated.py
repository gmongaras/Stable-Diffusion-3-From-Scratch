import torch
import torchvision
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader, Dataset
from scipy.linalg import sqrtm
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

# Directory containing subfolders [0, 999] with generated images
generated_images_path = "./output/softmax_4GPU_bs130_250Ksteps_1024dim_model_250000s.pkl/"

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1.0)  # Map to range [-1, 1]
])

# Custom dataset for a directory of images
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

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
        for images in tqdm(loader, desc="Calculating Activations"):
            images = images.cuda()

            # Save image samples
            for i, img in enumerate(images):
                img_name = f"{i}.png"
                torchvision.utils.save_image(img, img_name, normalize=True)

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

# Load precomputed real statistics (mu and sigma for each class)
real_statistics_path = "FID_results/Imagenet"
# real_stats = {
#     int(class_name.replace("_mu.npy", "")): {
#         "mu": np.load(os.path.join(real_statistics_path, class_name)),
#         "sigma": np.load(os.path.join(real_statistics_path, class_name).replace("mu", "sigma"))
#     }
#     for class_name in os.listdir(real_statistics_path) if class_name.endswith("_mu.npy")
# }

# Calculate FID for each class
fid_results = {}
for class_id in tqdm(range(1000), desc="Processing Classes"):
    class_folder = os.path.join(generated_images_path, str(class_id))
    # if not os.path.exists(class_folder) or class_id not in real_stats:
    #     print(f"Skipping class {class_id}: folder not found or no real statistics.")
    #     continue
    
    # Create dataset and loader for the generated images of this class
    dataset = ImageFolderDataset(class_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=64, pin_memory=True, drop_last=False, num_workers=10)
    
    # Calculate activations and statistics
    activations = calculate_activations(loader, inception)
    mu_gen, sigma_gen = calculate_statistics(activations)
    
    # Real statistics for the current class
    mu_real = np.load(os.path.join(real_statistics_path, str(class_id) + "_mu.npy"))
    sigma_real = np.load(os.path.join(real_statistics_path, str(class_id) + "_sigma.npy"))
    
    # Compute FID
    fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    fid_results[class_id] = fid

    print(f"Class {class_id}: FID = {fid}")

# Save the FID results to a file
os.makedirs("FID_results/generated", exist_ok=True)
np.save("FID_results/generated/fid_results.npy", fid_results)

# Print results
print("FID Results:")
for class_id, fid in fid_results.items():
    print(f"Class {class_id}: FID = {fid}")
