import torch
from models.diff_model import diff_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import click
import random






@click.command()

# Required
@click.option("--loadDir", "loadDir", type=str, help="Location of the models to load in.", required=True)
@click.option("--loadFile", "loadFile", type=str, help="Name of the .pkl model file to load in. Ex: model_358e_450000s.pkl", required=True)
@click.option("--loadDefFile", "loadDefFile", type=str, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl", required=True)

# Generation parameters
@click.option("--num_steps", "num_steps", type=int, default=10, help="Number of steps to generate an image", required=False)
@click.option("--device", "device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
@click.option("--guidance", "w", type=float, default=4, help="Classifier guidance scale which must be >= 0. The higher the value, the better the image quality, but the lower the image diversity.", required=False)
@click.option("--class_label", "class_label", type=int, default=0, help="0-indexed class value. Use -1 for a random class and any other class value >= 0 for the other classes. FOr imagenet, the class value range from 0 to 999 and can be found in data/class_information.txt", required=False)
@click.option("--sampler", "sampler", type=str, default="euler a", help="Sampler to use for generation.", required=False)
@click.option("--seed", "seed", type=int, default=-1, help="Seed for the random number generator.", required=False)
@click.option("--batch_size", "batch_size", type=int, default=2, help="Batch size for generation", required=False)

# Output parameters
@click.option("--out_imgname", "out_imgname", type=str, default="fig.png", help="Name of the file to save the output image to.", required=False)
@click.option("--out_gifname", "out_gifname", type=str, default="diffusion.gif", help="Name of the file to save the output image to.", required=False)
@click.option("--gif_fps", "gif_fps", type=int, default=10, help="FPS for the output gif.", required=False)

def infer(
    loadDir: str,
    loadFile: str,
    loadDefFile: str,

    num_steps: int,
    device: str,
    w: float,
    class_label: int,
    sampler: str,
    seed: int,
    batch_size: int,

    out_imgname: str,
    out_gifname: str,
    gif_fps: int
    ):

    
    
    
    
    
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

    # Create generator seed
    if seed != -1:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    # If class_label is -1, then sample a random class
    if class_label == -1:
        class_label = random.randint(0, num_classes-1)
        print(f"Random class selected: {class_label}")
    
    # Sample the model
    noise, imgs = model.sample_imgs(batch_size, num_steps, class_label, w, True, True, sampler, generator)
            
    # Convert the sample image to 0->255
    # and show it
    plt.close('all')
    plt.axis('off')
    noise = (((noise + 1)/2)*255)
    noise = torch.clamp(noise.cpu().detach().int(), 0, 255)
    for i, img in enumerate(noise):
        plt.imshow(img.permute(1, 2, 0))
        plt.savefig(f"{out_imgname}_{i}.png", bbox_inches='tight', pad_inches=0, )

    # Image evolution gif
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_axis_off()
    for i in range(0, len(imgs)):
        imgs[i] = (((imgs[i] + 1)/2)*255)
        imgs[i] = torch.clamp(imgs[i].cpu().detach().int(), 0, 255).permute(1, 2, 0)
        title = plt.text(imgs[i].shape[0]//2, -5, f"t = {i}", ha='center')
        imgs[i] = [plt.imshow(imgs[i], animated=True), title]
    animate = animation.ArtistAnimation(fig, imgs, interval=1, blit=True, repeat_delay=1000)
    animate.save(out_gifname, writer=animation.PillowWriter(fps=gif_fps))
    
    
    
    
    
if __name__ == '__main__':
    infer()