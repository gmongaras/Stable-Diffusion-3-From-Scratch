# Realtive import
import sys
sys.path.append('../helpers')
sys.path.append('../blocks')
sys.path.append("./")

import numpy as np
import torch
from torch import nn
try:
    from src.blocks.PositionalEncoding import PositionalEncoding
    from src.blocks.Transformer_Block_Dual import Transformer_Block_Dual
    from src.blocks.Transformer_Block_Single import Transformer_Block_Single
    from src.blocks.patchify import patchify, unpatchify
    from src.blocks.Norm import Norm
    from src.blocks.ImagePositionalEncoding import PatchEmbed, PatchEmbedAttn
    from src.blocks.TextPositionalEncoding import TextPositionalEncoding
    from src.helpers.VAE_T5_CLIP_inference import VAE_T5_CLIP_inference
except ModuleNotFoundError:
    from ..blocks.PositionalEncoding import PositionalEncoding
import os
import json
from tqdm import tqdm






class ImagePositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to a tensor.

    Args:
        d_model (int): The embedding dimension (size of the last dimension of `x_t`).
        max_len (int): The maximum length of the sequences. Default is 5000.

    Usage:
        pe = PositionalEncoding(d_model=d)
        x_t = pe(x_t)
    """
    def __init__(self, d_model, max_len=5000):
        super(ImagePositionalEncoding, self).__init__()

        # Create a positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.tensor(10000.0)) / d_model))  # Shape: (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Registers `pe` as a buffer (not a parameter)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, S, d_model)

        Returns:
            Tensor: The input tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x







class diff_model(nn.Module):
    # inCh - Number of input channels in the input batch
    # num_classes - Number of classes to condition the model on
    # patch_size - Size of the patches to embed
    # dim - Dimension to embed each patch
    # c_dim - Dimension to embed the class info
    # hidden_scale - Multiplier to scale in the MLP
    # num_heads - Number of heads in the attention blocks
    # attn_type - Type of attention to use in the transformer ("softmax" or "cosine")
    # num_blocks - Number of blocks in the transformer
    # device - Device to put the model on (gpu or cpu)
    # start_step - Step to start on. Doesn't do much besides 
    #               change the name of the saved output file
    def __init__(self, inCh, class_dim, patch_size, dim, hidden_scale, num_heads, attn_type, num_blocks, device, positional_encoding, checkpoint_MLP=True, start_step=0, wandb_id=None):
        super(diff_model, self).__init__()
        
        self.inCh = inCh
        self.class_dim = class_dim
        self.patch_size = patch_size
        self.start_step = start_step
        self.wandb_id = wandb_id
        
        # Important default parameters
        self.defaults = {
            "inCh": inCh,
            "class_dim": class_dim,
            "patch_size": patch_size,
            "dim": dim,
            "hidden_scale": hidden_scale,
            "num_heads": num_heads,
            "attn_type": attn_type,
            "num_blocks": num_blocks,
            "positional_encoding": positional_encoding,
            "device": "cpu",
            "start_step": start_step,
            "wandb_id": wandb_id,
        }
        
        # Convert the device to a torch device
        if type(device) is str:
            if device.lower() == "gpu":
                if torch.cuda.is_available():
                    dev = device.lower()
                    try:
                        local_rank = int(os.environ['LOCAL_RANK'])
                    except KeyError:
                        local_rank = 0
                    device = torch.device(f"cuda:{local_rank}")
                else:
                    dev = "cpu"
                    print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                    device = torch.device('cpu')
            else:
                dev = "cpu"
                device = torch.device('cpu')
            self.device = device
            self.dev = dev
        else:
            self.device = device
            self.dev = "cpu" if device.type == "cpu" else "gpu"
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Transformer_Block_Dual(dim, c_dim=dim, hidden_scale=hidden_scale, num_heads=num_heads, attn_type=attn_type, positional_encoding=positional_encoding, checkpoint_MLP=checkpoint_MLP, layer_idx=i, last=(i==num_blocks-1)).to(device)
            for i in range(num_blocks)
        ])
            
        # Used to embed the values of t so the model can use it
        self.t_emb = PositionalEncoding(dim, device=device).to(device)
        self.t_emb2 = nn.Linear(dim, dim, bias=False).to(device)

        # Input conditional MLP. Used to embed c_pooled
        self.cond_MLP = nn.Linear(self.class_dim, dim).to(device)

        # Used to embed the values of c so the model can use it
        # self.c_pos_enc = TextPositionalEncoding(4096, 154, learnable=False).to(device)
        self.c_proj = nn.Linear(4096, dim, bias=False).to(device)
        
        # Patch embedding (inCh*P*P --> dim)
        # self.patch_emb = nn.Linear(inCh*patch_size*patch_size, dim)
        self.patch_emb = nn.Linear(dim, dim).to(device)
        # Positional encodings for the patches
        # self.pos_enc = ImagePositionalEncoding(dim)
        # self.pos_enc = PatchEmbedAttn(
        #     height=256, 
        #     width=256, 
        #     patch_size=self.patch_size, 
        #     in_channels=inCh,
        #     embed_dim=dim,
        #     layer_norm=False, 
        #     flatten=True, 
        #     bias=False, 
        #     interpolation_scale=1, 
        #     pos_embed_type="sincos", 
        #     pos_embed_max_size=256
        # )
        self.pos_enc = PatchEmbed(
            height=256, 
            width=256, 
            patch_size=self.patch_size, 
            in_channels=inCh,
            embed_dim=dim,
            layer_norm=False, 
            flatten=True, 
            bias=False, 
            interpolation_scale=1, 
            pos_embed_type=positional_encoding,
            pos_embed_max_size=256
        ).to(device)
        # Output norm
        self.out_norm = Norm(dim, dim).to(device)
        # Output projection
        self.out_proj = nn.Linear(dim, inCh*patch_size*patch_size).to(device)

        
            
            
    # Unsqueezing n times along the given dim.
    # Note: dim can be 0 or -1
    def unsqueeze(self, X, dim, n):
        if dim == 0:
            return X.reshape(n*(1,) + X.shape)
        else:
            return X.reshape(X.shape + (1,)*n)
    
        
    # Used to noise a batch of images by t timesteps
    # Inputs:
    #   X - Batch of images of shape (N, C, L, W)
    #   t - Batch of t values of shape (N)
    # Outputs:
    #   Batch of noised images of shape (N, C, L, W)
    #   Batch of noised images of shape (N, C, L, W)
    #   Noise added to the images of shape (N, C, L, W)
    def noise_batch(self, X, t):
        # Ensure the data is on the correct device
        X = X.to(self.device)
        t = t.to(self.device)[:, None, None, None]

        # Sample gaussian noise
        epsilon = torch.randn_like(X, device=self.device)
        
        # Recfitied flow
        X_t = (1-t)*X + t*epsilon
        
        # Noise the images
        return X_t, epsilon



    # Used to convert a batch of noise predictions to
    # a batch of mean predictions
    # Inputs:
    #   epsilon - The epsilon value for the mean of shape (N, C, L, W)
    #   x_t - The image to unoise of shape (N, C, L, W)
    #   t - A batch of t values for the beta schedulers of shape (N)
    #   corrected - True to calculate the corrected mean that doesn't
    #               go outside the bounds. False otherwise.
    # Outputs:
    #   A tensor of shape (N, C, L, W) representing the mean of the
    #     unnoised image
    def noise_to_mean(self, epsilon, x_t, t, corrected=True):
        # Note: Corrected function from the following:
        # https://github.com/hojonathanho/diffusion/issues/5

        
        # Get the beta and a values for the batch of t values
        beta_t = self.scheduler.sample_beta_t(t)
        sqrt_a_t = self.scheduler.sample_sqrt_a_t(t)
        a_bar_t = self.scheduler.sample_a_bar_t(t)
        sqrt_a_bar_t = self.scheduler.sample_sqrt_a_bar_t(t)
        sqrt_1_minus_a_bar_t = self.scheduler.sample_sqrt_1_minus_a_bar_t(t)
        a_bar_t1 = self.scheduler.sample_a_bar_t1(t)
        sqrt_a_bar_t1 = self.scheduler.sample_sqrt_a_bar_t1(t)


        if len(t.shape) == 1:
            t = self.unsqueeze(t, -1, 3)


        # Calculate the uncorrected mean
        if corrected == False:
            return (1/sqrt_a_t)*(x_t - (beta_t/sqrt_1_minus_a_bar_t)*epsilon)


        # Calculate the corrected mean and return it
        mean = torch.where(t == 0,
            # When t is 0, normal without correction
            (1/sqrt_a_t)*(x_t - (beta_t/sqrt_1_minus_a_bar_t)*epsilon),

            # When t is not 0, special with correction
            (sqrt_a_bar_t1*beta_t)/(1-a_bar_t) * \
                torch.clamp( (1/sqrt_a_bar_t)*x_t - (sqrt_1_minus_a_bar_t/sqrt_a_bar_t)*epsilon, -1, 1 ) + \
                (((1-a_bar_t1)*sqrt_a_t)/(1-a_bar_t))*x_t
        )
        return mean
    


    def encode_text_to_block(self, text):
        # L4 encodings
        self.CLIPL4_processor(text=["a photo of a cat", "a photo of a dog"], images=None, return_tensors="pt", padding=True, device=self.device)

    

    def load_text_encoders(self):
        self.text_encoders = VAE_T5_CLIP_inference(self.device)
    
    
    
    # Input:
    #   x_t - Batch of images of shape (B, C, L, W)
    #   t - Batch of t values of shape (N) or a single t value. Note
    #       that this t value represents the timestep the model is currently at.
    #   c - Batch of class values of shape (N, 154, 4096)
    #   c_pooled- Batch of pooled class values of shape (N, 2048)
    #   nullL4 - Binary tensor of shape (N) where a 1 represents a null representation for the L4 encoder
    #   nullG14 - Binary tensor of shape (N) where a 1 represents a null representation for the G14 encoder
    #   nullT5 - Binary tensor of shape (N) where a 1 represents a null representation for the T5 encoder
    # Outputs:
    #   noise - Batch of noise predictions of shape (B, C, L, W)
    #   v - Batch of v predictions of shape (B, C, L, W)
    def forward(self, x_t, t, c, c_pooled, nullL4=None, nullG14=None, nullT5=None):
        # Ensure the data is on the correct device
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        c = c.to(self.device)
        c_pooled = c_pooled.to(self.device)
        nullL4 = nullL4.to(self.device)
        nullG14 = nullG14.to(self.device)
        nullT5 = nullT5.to(self.device)


        

        # Handling null class values
        if type(nullL4) != type(None):
            # Mask the upper left corner of the text matrix
            c[nullL4, :77, :768] *= 0
            # Mask the upper part of the pooled vector
            c_pooled[nullL4, :768] *= 0
        if type(nullG14) != type(None):
            # Mask the middle left of the text matrix
            c[nullG14, :77, 768:2048] *= 0
            # Mask the lower part of the pooled vector
            c_pooled[nullG14, 768:2048] *= 0
        if type(nullT5) != type(None):
            # Mask the right of the text matrix
            c[nullT5, 77:, :] *= 0
            # No pooled part to mask




        # Make sure t is in the correct form
        if t != None:
            if type(t) == int or type(t) == float:
                t = torch.tensor(t).repeat(x_t.shape[0]).to(torch.long)
            elif type(t) == list and type(t[0]) == int:
                t = torch.tensor(t).to(torch.long)
            elif type(t) == torch.Tensor:
                if len(t.shape) == 0:
                    t = t.repeat(x_t.shape[0]).to(torch.long)
            else:
                print(f"t values must either be a scalar, list of scalars, or a tensor of scalars, not type: {type(t)}")
                return
            
            # Encode the timesteps
            if len(t.shape) == 1:
                t = self.t_emb2(self.t_emb(t))


        # Embed the pooled class info
        c_pooled = self.cond_MLP(c_pooled.to(self.cond_MLP.weight.dtype))
                
        # Combine the class and time embeddings
        y = t.to(c_pooled.dtype) + c_pooled
            
        # Original shape of the images
        orig_shape = x_t.shape
        
        # Patchify the input images
        # x_t = patchify(x_t, (self.patch_size, self.patch_size))

        # Add positional encodings to the text and projecct to the embedding dim
        # No positiona lencoding so that tokens don't have a position bias
        c = self.c_proj(c.to(self.c_proj.weight.dtype))

        # Patchify and add the positional encoding
        x_t = self.pos_enc(x_t.to(c.dtype))
        
        # Send the patches through the patch embedding
        x_t = self.patch_emb(x_t)

        # Send the patches through the transformer blocks
        for i, block in enumerate(self.blocks):
            x_t, c = block(x_t, c, y)
            
        # Send the output through the output projection
        x_t = self.out_proj(self.out_norm(x_t, y))
        
        # Unpatchify the images
        x_t = unpatchify(x_t, (self.patch_size, self.patch_size), orig_shape[-2:])
        
        return x_t


    # Sample a batch of generated samples from the model
    # Params:
    #   batchSize - Number of images to generate in parallel
    #   class_label - (optional and only used if the model uses class info) 
    #                 Class we want the model to generate
    #                 Use -1 to generate without a class
    #   w - (optional and only used if the model uses class info) 
    #       Classifier guidance scale factor. Use 0 for no classifier guidance.
    #   save_intermediate - Return intermediate generation states
    #                       to create a gif along with the image?
    #   use_tqdm - Show a progress bar or not
    #   unreduce - True to unreduce the image to the range [0, 255],
    #              False to keep the image in the range [-1, 1]
    #   corrected - True to put a limit on generation. False to not restrain generation
    # Outputs:
    #   output - Output images of shape (N, C, L, W)
    #   imgs - (only if save_intermediate=True) list of iternediate
    #          outputs for the first image i the batch of shape (steps, C, L, W)
    @torch.no_grad()
    def sample_imgs(self, batchSize, num_steps, text_input, cfg_scale=0.0, save_intermediate=False, use_tqdm=False, sampler="euler a", generator=None):
        use_vae = True
        
        # Make sure the model is in eval mode
        self.eval()

        # The initial image is pure noise
        h = w = 256
        output = torch.randn((batchSize, 4 if use_vae else 3, h//8 if use_vae else h, w//8 if use_vae else w), generator=generator).to(self.device)
        eps = output.clone()

        # Encode the text
        text_hidden, text_pooled = self.text_encoders.text_to_embedding(text_input)

        # Put class label on device and add null label for CFG
        nullCls = (torch.tensor([0]*batchSize+[1]*batchSize).bool().to(self.device))
        text_hidden = (text_hidden.repeat(2*batchSize, 1, 1).to(self.device))
        text_pooled = (text_pooled.repeat(2*batchSize, 1).to(self.device))

        imgs = []

        # # Iterate from t=1 to t=0 for a total of num_steps steps
        # for i, t in enumerate(tqdm(range(num_steps), total=num_steps)) if use_tqdm else enumerate(range(num_steps)):
        #     # t starts at 1 and ends at 0
        #     t = torch.tensor([1 - (t / num_steps)]).repeat(2*batchSize).to(self.device)

        #     # Get model velocity prediction. Twice for CFG
        #     velocity = self.forward(output.repeat(2, 1, 1, 1), t, class_label, nullCls)

        #     # Get CFG output
        #     velocity = (1 + cfg_scale) * velocity[:batchSize] - cfg_scale * velocity[batchSize:]

        #     # We have v_t = epsilon_t - x_t and x_t = (1-t)*x_0 + t*epsilon_t
        #     # v_t represents how x changes as t increases.
        #     # To move from x_1=epsilon_1 to x_0=x_0, we need to
        #     # move in the opposite direction of v_t.
        #     # So we move in the direction of -v_t
        #     # dx/dt * dt = dx, the change in x we want
        timesteps = torch.linspace(1, 0 + (1.0 / num_steps), num_steps).to(self.device)  # Linear schedule (can use cosine)
        for i, t in enumerate(tqdm(timesteps, total=num_steps) if use_tqdm else timesteps):
            # Dynamic CFG scale
            dynamic = False
            if dynamic:
                cfg_scale_dynamic = cfg_scale * (t.item() ** 2)
            else:
                cfg_scale_dynamic = cfg_scale

            t = t.repeat(2*batchSize).to(self.device)

            # Predict velocity twice for CFG
            velocity = self.forward(output.repeat(2, 1, 1, 1), t, text_hidden, text_pooled, nullCls, nullCls, nullCls)
            velocity = (1 + cfg_scale_dynamic) * velocity[:batchSize] - cfg_scale_dynamic * velocity[batchSize:]

            dt = 1 / num_steps  # Step size

            # Choose sampler
            if sampler == "euler":
                # Euler method
                output = output - velocity * dt

            elif sampler == "euler_stochastic":
                # Calculate sigma (noise scale) based on timestep
                # This sigma funciton  is highest in the middle of sampling Reduces to near zero at the end
                sigma = (t * (1 - t) / (1 - t + 0.008))[:batchSize, None, None, None]  # You can adjust the 0.008 constant
                # # Linear schedule
                # sigma = t[:batchSize, None, None, None]
                # # Cosine schedule
                # sigma = torch.cos(t * torch.pi / 2)[:batchSize, None, None, None]
                # # Exponential schedule
                # sigma = torch.exp(-5 * (1-t))[:batchSize, None, None, None]
                
                # Generate random noise scaled by sigma
                noise = torch.randn(velocity.shape, generator=generator).to(output.device)
                # noise = torch.randn_like(velocity, generator=generator).to(output.device)
                
                # Update output using Euler a step
                output = output - velocity * dt + sigma * noise * np.sqrt(dt)

            elif sampler == "heun":
                # Heun's method (2nd order solver)
                velocity_1 = velocity
                x_pred = output - velocity_1 * dt  # Euler step prediction

                # Next time step
                t_next = t - dt
                velocity_2 = self.forward(x_pred.repeat(2, 1, 1, 1), t_next, text_hidden, text_pooled, nullCls, nullCls, nullCls)
                velocity_2 = (1 + cfg_scale_dynamic) * velocity_2[:batchSize] - cfg_scale_dynamic * velocity_2[batchSize:]

                # Correct step using average velocity
                output = output - (dt / 2) * (velocity_1 + velocity_2)

            else:
                raise ValueError("Invalid sampler specified. Choose 'euler', 'euler_stochastic', or 'heun'.")
            
            if save_intermediate:
                if use_vae:
                    imgs.append(self.text_encoders.VAE.decode(output.to(self.text_encoders.VAE.dtype) / self.text_encoders.VAE.config.scaling_factor).sample.clamp(-1, 1)[0].float().cpu().detach())
                else:
                    imgs.append(output[0].cpu().detach())
        
        if save_intermediate:
            if use_vae:
                imgs.append(self.text_encoders.VAE.decode(output.to(self.text_encoders.VAE.dtype) / self.text_encoders.VAE.config.scaling_factor).sample.clamp(-1, 1)[0].float().cpu().detach())
            else:
                imgs.append(output[0].cpu().detach())

        output = self.text_encoders.VAE.decode(output.to(self.text_encoders.VAE.dtype) / self.text_encoders.VAE.config.scaling_factor).sample.clamp(-1, 1).float() if use_vae else output

        return (output, imgs) if save_intermediate else output


    
    # Save the model
    # saveDir - Directory to save the model state to
    # optimizer (optional) - Optimizer object to save the state of
    # scheduler (optional) - Scheduler object to save the state of
    # step (optional) - Current step of the model (helps when loading state)
    def saveModel(self, saveDir, optimizer=None, scheduler=None, grad_scalar=None, step=None):
        # Craft the save string
        saveFile = "model"
        optimFile = "optim"
        schedulerFile = "scheduler"
        scalarFile = "scaler"
        saveDefFile = "model_params"
        if step:
            saveFile += f"_{step}s"
            optimFile += f"_{step}s"
            schedulerFile += f"_{step}s"
            scalarFile += f"_{step}s"
            saveDefFile += f"_{step}s"
        saveFile += ".pkl"
        optimFile += ".pkl"
        schedulerFile += ".pkl"
        scalarFile += ".pkl"
        saveDefFile += ".json"

        # Change step state if given
        if step:
            self.defaults["start_step"] = step

        # Update wandb id
        self.defaults["wandb_id"] = self.wandb_id
        
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        # Save the model and the optimizer
        torch.save(self.state_dict(), saveDir + os.sep + saveFile)
        if optimizer:
            torch.save(optimizer.state_dict(), saveDir + os.sep + optimFile)
        if scheduler:
            torch.save(scheduler.state_dict(), saveDir + os.sep + schedulerFile)
        if grad_scalar:
            torch.save(grad_scalar.state_dict(), saveDir + os.sep + scalarFile)

        # Save the defaults
        with open(saveDir + os.sep + saveDefFile, "w") as f:
            json.dump(self.defaults, f)
    
    
    # Load the model
    # loadDir - Directory to load the model from
    # loadFile - Pytorch model file to load in
    # loadDefFile (Optional) - Defaults file to load in
    def loadModel(self, loadDir, loadFile, loadDefFile=None, wandb_id=None):
        if loadDefFile:
            device_ = self.device
            dev_ = self.dev

            # Load in the defaults
            with open(loadDir + os.sep + loadDefFile, "r") as f:
                self.defaults = json.load(f)
            D = self.defaults
            if "positional_encoding" not in D:
                D["positional_encoding"] = "absolute"

            # Reinitialize the model with the new defaults
            self.__init__(**D)
            self.to(device_)
            self.device = device_
            self.dev = dev_

            # Load the model state
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device), strict=True)

        else:
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device), strict=True)