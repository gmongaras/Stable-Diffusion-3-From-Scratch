# Realtive import
import sys
sys.path.append('./src')

import torch
from models.diff_model import diff_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import click
import os
from tqdm import tqdm






@click.command()

# Required
@click.option("--loadDir", "loadDir", type=str, help="Location of the models to load in.", required=True)
@click.option("--loadFile", "loadFile", type=str, help="Name of the .pkl model file to load in. Ex: model_358e_450000s.pkl", required=True)
@click.option("--loadDefFile", "loadDefFile", type=str, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl", required=True)

# Generation parameters
@click.option("--num_steps", "num_steps", type=int, default=10, help="Number of steps to generate an image", required=False)
@click.option("--device", "device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
@click.option("--guidance", "w", type=float, default=4, help="Classifier guidance scale which must be >= 0. The higher the value, the better the image quality, but the lower the image diversity.", required=False)
@click.option("--num_per_class", "num_per_class", type=int, default=128, help="Number of images to generate per class.", required=False)
@click.option("--batch_size", "batch_size", type=int, default=32, help="Batch size for generation", required=False)
@click.option("--sampler", "sampler", type=str, default="euler a", help="Sampler to use for generation.", required=False)
@click.option("--seed", "seed", type=int, default=-1, help="Seed for the random number generator.", required=False)
@click.option("--start_class", "start_class", type=int, default=0, help="Class to start generating images from.", required=False)


def generate_images(
    loadDir: str,
    loadFile: str,
    loadDefFile: str,

    num_steps: int,
    device: str,
    w: float,
    num_per_class: int,
    batch_size: int,
    sampler: str,
    seed: int,
    start_class: int
    ):

    
    
    assert num_per_class % batch_size == 0, "num_per_class must be divisible by batch_size"
    
    
    ### Model Creation

    # Create a dummy model
    inCh = 4
    num_classes = 1000
    patch_size = 2
    dim = 1024
    c_dim = 512
    hidden_scale = 2.0
    num_heads = 8
    attn_type = "cosine"
    num_blocks = 20
    device = "gpu"

    model = diff_model(
        inCh=inCh,
        num_classes=num_classes,
        patch_size=patch_size,
        dim=dim,
        c_dim=c_dim,
        hidden_scale=hidden_scale,
        num_heads=num_heads,
        attn_type=attn_type,
        num_blocks=num_blocks,
        device=device,
    )
    
    # Load in the model weights
    model.loadModel(loadDir, loadFile, loadDefFile)

    # Load on device
    if device == "gpu":
        model = model.cuda()
    else:
        model = model
    model.device = model.c_emb.weight.device

    # Create the output directory
    s = loadDir.split("/")[-1]
    base_dir = f"output/{s}_{loadFile}/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Iterate over all the classes
    for cls in tqdm(range(start_class, num_classes)):
        # Create generator seed for this class
        if seed != -1:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        # Generate all the images
        for i in range(0, num_per_class, batch_size):
            # Make the class directory
            class_dir = f"{base_dir}/{cls}/"
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            # Sample the model
            noise = model.sample_imgs(batch_size, num_steps, cls, w, False, False, sampler, generator)
                
            # Convert the sample image to 0->255
            # and show it
            plt.close('all')
            plt.axis('off')
            noise = (((noise + 1)/2)*255)
            noise = torch.clamp(noise.cpu().detach().int(), 0, 255)
            for j, img in enumerate(noise):
                img_name = f"{class_dir}/{i+j}.png"
                plt.imshow(img.permute(1, 2, 0))
                plt.savefig(img_name, bbox_inches='tight', pad_inches=0, )
        
    
    
    
    
if __name__ == '__main__':
    generate_images()