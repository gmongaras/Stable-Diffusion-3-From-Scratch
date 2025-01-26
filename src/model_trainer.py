import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from tqdm import tqdm
import copy
import time

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

try:
    from helpers.multi_gpu_helpers import is_main_process
    from helpers.TimeSampler import TimeSampler
    from helpers.VAE_T5_CLIP import VAE_T5_CLIP
except ModuleNotFoundError:
    from .helpers.multi_gpu_helpers import is_main_process
    from .helpers.TimeSampler import TimeSampler
    from .helpers.VAE_T5_CLIP import VAE_T5_CLIP


cpu = torch.device('cpu')



from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup

def get_scheduler(optimizer, num_warmup_steps, num_training_steps, use_lr_scheduler):
    """
    Creates a cosine scheduler with warmup
    """
    if use_lr_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5  # Standard half-cycle cosine
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    return scheduler




def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    

    # Try the nccl backend
    try:
        dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
                rank=rank)
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
                backend="gloo",
                init_method=dist_url,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
                rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()




# Distributed without loader GPUs in the group
def init_distributed_no_loaders(loader_gpus):
    world_size = dist.get_world_size()
    ranks_to_include = [rank for rank in range(world_size) if rank not in loader_gpus]

    # Create a new group with ranks other than 0
    subgroup = dist.new_group(ranks=ranks_to_include)
    return subgroup




# Trains a diffusion model
class model_trainer():
    # diff_model - A diffusion model to train
    # batchSize - Batch size to train the model with
    # numSteps - Number of steps to breakup the batchSize into. Instead
    #            of taking 1 massive step where the whole batch is loaded into
    #            memory, the batchSize is broken up into sizes of
    #            batchSize//numSteps so that it can fit into memory. Mathematically,
    #            the update will be the same, as a single batch update, but
    #            the update is distributed across smaller updates to fit into memory.
    # totalSteps - Number of steps to train the model for
    # lr - Learning rate of the model optimizer
    # device - Device to put the model and data on (gpu or cpu)
    # saveDir - Directory to save the model to
    # numSaveSteps - Number of steps until saving the models
    # use_importance - True to use importance sampling to sample values of t,
    #                  False to use uniform sampling.
    # p_uncond - Probability of training on a null class (only used if class info is used)
    # load_into_mem - True to load all data into memory first, False to load from disk as needed
    # optimFile - Optional name of optimizer to load in
    # schedulerFile - Optional name of scheduler to load in
    def __init__(self, 
            diff_model, 
            batchSize, 
            numSteps, 
            totalSteps, 
            lr, 
            ema_update_freq, 
            ema_decay, 
            warmup_steps,
            use_lr_scheduler,
            device, 
            saveDir, 
            numSaveSteps, 
            null_prob_L4=0.464, 
            null_prob_G14=0.464, 
            null_prob_T5=0.464,
            optimFile=None, 
            schedulerFile=None, 
            scalerFile=None, 
            use_amp=True, 
            wandb_name=None, 
            log_steps=10,
            loader_to_model_gpu=None):
        # Saved info
        self.batchSize = batchSize//numSteps
        self.numSteps = numSteps
        self.totalSteps = totalSteps
        self.ema_update_freq = ema_update_freq
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.use_lr_scheduler = use_lr_scheduler
        self.saveDir = saveDir
        self.numSaveSteps = numSaveSteps
        self.null_prob_L4 = null_prob_L4
        self.null_prob_G14 = null_prob_G14
        self.null_prob_T5 = null_prob_T5
        self.use_amp = use_amp
        self.wandb_name = wandb_name
        self.log_steps = log_steps
        self.loader_to_model_gpu = loader_to_model_gpu


        # # The first GPUs are the data loader GPUs
        # self.loader_gpus = [i for i in range(self.num_loader_gpus)]
        # self.model_gpus = [i for i in range(self.num_loader_gpus, self.num_loader_gpus + self.num_loader_gpus + self.num_model_gpus_per_loader)]


        # Map from dataloader gpu num to model gpu num
        # self.loader_to_model_gpu = {loader_gpu: [self.num_loader_gpus+(loader_gpu*self.num_model_gpus_per_loader) + model_gpu for model_gpu in range(0, self.num_model_gpus_per_loader)] for loader_gpu in self.loader_gpus}
        self.model_gpus = sum(list(self.loader_to_model_gpu.values()), [])
        self.loader_gpus = list(self.loader_to_model_gpu.keys())
        # Map from model gpu num to dataloader gpu num
        self.model_to_loader_gpu = dict()
        for loader_gpu in self.loader_to_model_gpu:
            for model_gpu in self.loader_to_model_gpu[loader_gpu]:
                self.model_to_loader_gpu[model_gpu] = loader_gpu

        # The world size must be equal to the number of (data loader gpus) * (number of model gpus per loader)
        # assert len(self.loader_gpus) + len(self.model_gpus) == int(os.environ['WORLD_SIZE']), "The number of loader gpus and model gpus must be equal to the number of GPUs available"

        
        # Convert the device to a torch device
        if device.lower() == "gpu":
            if torch.cuda.is_available():
                dev = device.lower()
                local_rank = int(os.environ['LOCAL_RANK'])
                rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
                device = torch.device(f"cuda:{local_rank}")

                self.rank = rank
                self.world_size = world_size
                self.local_rank = local_rank
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = "cpu"
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        # Put the model on the desired device
        if dev != "cpu":
            # Initialize the environment
            init_distributed()

            # Get the subgroup
            self.subgroup = init_distributed_no_loaders(self.loader_gpus)

            # Do not create a model on the loader gpus
            if rank not in self.loader_gpus:
                self.model = DDP(diff_model.cuda(local_rank), device_ids=[local_rank], process_group=self.subgroup, broadcast_buffers=False, find_unused_parameters=False)
        else:
            self.model = diff_model.cpu()


        
        
        # Logging
        if is_main_process(self.subgroup):
            print(f"Total GPUs: {self.world_size}")
            print(f"Model GPUs: {self.model_gpus}")
            print(f"Loader GPUs: {self.loader_gpus}")
            print(f"Model to loader mapping: {self.model_to_loader_gpu}")
            print(f"Loader to model mapping: {self.loader_to_model_gpu}")







        # Data loader GPUs
        if rank in self.loader_gpus:
            # Load in the VAE and T5 models onto the first device
            self.VAE_T5_CLIP = VAE_T5_CLIP(batchSize, torch.device(f"cuda:{local_rank}"), loader_to_model_gpu=self.loader_to_model_gpu, rank=rank, world_size=world_size, num_batches=len(self.loader_to_model_gpu[rank]))
        dist.barrier(group=self.subgroup)


        # Get the loader gpu that the model gpu is mapped to
        self.loader_gpu = self.model_to_loader_gpu[self.rank]


        # EMA model on CPU to save GPU memory
        self.ema_model_cpu = copy.deepcopy(self.model.module).cpu()
        self.ema_model_cpu.eval()
        
        # Optimizer
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-8, weight_decay=0.01, betas=(0.9, 0.999))

        # LR Scheduler
        self.scheduler = get_scheduler(self.optim, num_warmup_steps=warmup_steps, num_training_steps=totalSteps, use_lr_scheduler=use_lr_scheduler)

        # Automatic mixed precision gradient scalar
        if self.use_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")
        else:
            self.grad_scaler = None

        # Load in optimizer paramters if they exist
        if optimFile:
            self.optim.load_state_dict(torch.load(optimFile, map_location=self.device))

        # Load in scheduler paramters if they exist
        if schedulerFile:
            self.scheduler.load_state_dict(torch.load(schedulerFile, map_location=self.device))

        # Load in scalar paramters if they exist
        if scalerFile:
            self.grad_scaler.load_state_dict(torch.load(scalerFile, map_location=self.device))

        # Load in states from the pretrained diffusion model
        self.wandb_id = self.model.wandb_id if dev == "cpu" else self.model.module.wandb_id
        self.start_step = self.model.start_step if dev == "cpu" else self.model.module.start_step

        # Used to sample timesteps
        self.time_sampler = TimeSampler(weighted=True)

        # Total params of the model
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Number of parameters in the model: {total_params:.2f}M")



    # Trains the model
    def train(self, ):
        # Put the model is train mode
        self.model.train()

        # Number of steps taken so far
        num_steps = self.start_step

        # Losses over steps
        self.losses_comb = np.array([])
        self.steps_list = np.array([])

        # Cumulative loss over the batch over each set of steps
        losses_comb_s = torch.tensor(0.0, requires_grad=False)

        batch_loss = 0

        # Initialize wandb run
        if is_main_process(self.subgroup):
            wandb.init(
                project="Cottention_Diffusion",
                name=self.wandb_name,
                notes=None, # May add notes later
                
                # Resume training if checkpoint exists
                resume="must" if self.wandb_id is not None else None,
                id=self.wandb_id,
            )
            wandb.watch(self.model, log_freq=self.log_steps)
            
            # Save wandb run id
            self.wandb_id = wandb.run.id
            self.model.wandb_id = self.wandb_id 
            if self.dev != "cpu":
                self.model.module.wandb_id = self.wandb_id

        # Sync all processes
        dist.barrier(self.subgroup)

        # Iterate over the desiered number of steps
        for step in range(num_steps, self.totalSteps):
            # for step, data in enumerate(tqdm(data_loader, initial=num_steps, total=self.totalSteps)):
            step = step + self.start_step

            # Get the data
            # data = self.VAE_T5_CLIP.load_data().to(self.device)
            # NOTE: We want to place the tensors on the local gpu but send via the global gpu.
            # dist.barrier(self.subgroup)
            batch_x_0 = torch.empty((self.batchSize, 4, 256//8, 256//8), dtype=torch.float16, device=self.device)
            batch_txt = torch.empty((self.batchSize, 154, 4096), dtype=torch.float16, device=self.device)
            batch_txt_pooled = torch.empty((self.batchSize, self.model.module.class_dim), dtype=torch.float16, device=self.device)
            request_flag = torch.tensor([1], device=f"cuda:{self.local_rank}") # Request signal
            # Send request flag
            dist.send(request_flag, dst=self.loader_gpu)
            # Get the data
            dist.recv(batch_x_0, src=self.loader_gpu)
            dist.recv(batch_txt, src=self.loader_gpu)
            dist.recv(batch_txt_pooled, src=self.loader_gpu)
            # dist.barrier(self.subgroup)
            # Increate the number of steps taken
            num_steps += 1
            
            # # Get a random value between 0 and 1
            # t_vals = torch.rand(batch_x_0.shape[0], device=batch_x_0.device)
            # Weighted timestep, still betwee 0 and 1
            t_vals = self.time_sampler(batch_x_0.shape[0])


            # Probability of each of the text embeddings being null
            probs_L4 = torch.rand(batch_x_0.shape[0])
            probs_G14 = torch.rand(batch_x_0.shape[0])
            probs_T5 = torch.rand(batch_x_0.shape[0])
            nullCls_L4 = torch.where(probs_L4 < self.null_prob_L4, 1, 0).to(torch.bool).to(self.device)
            nullCls_G14 = torch.where(probs_G14 < self.null_prob_G14, 1, 0).to(torch.bool).to(self.device)
            nullCls_T5 = torch.where(probs_T5 < self.null_prob_T5, 1, 0).to(torch.bool).to(self.device)
            

            # Noise the batch to time t
            with torch.no_grad():
                if self.dev == "cpu":
                    batch_x_t, epsilon_t = self.model.noise_batch(batch_x_0, t_vals)
                else:
                    batch_x_t, epsilon_t = self.model.module.noise_batch(batch_x_0, t_vals)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                # Send the noised data through the model to get the predicted noise
                v_pred = self.model(batch_x_t.detach(), t_vals,  batch_txt, batch_txt_pooled, nullCls_L4, nullCls_G14, nullCls_T5)

                # The label is the velocity: 
                # v_t = alpha_t' * x + sigma_t' * epsilon_t
                # v_t = (1-t)' * x + (t)' * epsilon_t
                # v_t = -x + epsilon_t
                # v_t = epsilon_t - x
                # labels = batch_x_0.to(epsilon_t.device) - epsilon_t
                labels = epsilon_t - batch_x_0.to(epsilon_t.device)

                # MSE between noise and predicted noise
                loss = nn.MSELoss(reduction="none")(v_pred, labels.detach()).flatten(1, -1)
                
                weigh_loss = False

                if weigh_loss:
                    def lognorm(t, m=0, s=1):
                        return (1/(s*np.sqrt(2*np.pi)))*(1/(t*(1-t)))*torch.exp(-(((torch.log(t/(1-t))-m)**2)/(2*(s**2))))

                    # Weight for rectified flows
                    weight = (t_vals / (1-t_vals)) * lognorm(t_vals, m=0.0, s=1.0)
                    
                    # Weighted loss
                    loss = (loss * weight[:, None].to(loss.device).detach()).mean()
                else:
                    loss = loss.mean()

            print(num_steps, loss)

            # Scale the loss to be consistent with the batch size. If the loss
            # isn't scaled, then the loss will be treated as an independent
            # batch for each step. If it is scaled by the step size, then the loss will
            # be treated as a part of a larger batchsize which is what we want
            # to acheive when using steps.
            loss = loss/self.numSteps

            # Backpropagate loss
            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            # Save the loss values
            losses_comb_s += loss.cpu().detach()
            batch_loss += loss.cpu().detach().item()

            # If the number of steps taken is a multiple of the number
            # of desired steps, update the models
            if num_steps%self.numSteps == 0:
                # Unscale gradients
                if self.use_amp:
                    self.grad_scaler.unscale_(self.optim)

                # # Clip gradients
                # if self.use_amp:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update the model using all losses over the steps
                if self.use_amp:
                    self.grad_scaler.step(self.optim)
                else:
                    self.optim.step()

                # Update scheduler
                self.scheduler.step(step)
                    
                # Step the gradient scaler
                if self.use_amp:
                    self.grad_scaler.update()
                            
                # Zero gradients 
                self.optim.zero_grad()

                # if is_main_process():
                #     print(f"step #{num_steps}   Latest loss estimate: {round(losses_comb_s.cpu().detach().item(), 6)}")

                # Log wandb
                if num_steps % self.log_steps == 0:
                    batch_loss = batch_loss/self.log_steps
                    
                    if is_main_process(self.subgroup):
                        wandb.log({
                            "loss": batch_loss,
                            "lr": self.optim.param_groups[0]['lr'],
                        },
                        step=num_steps)
                    
                    batch_loss = 0

                # Save the loss values
                self.losses_comb = np.append(self.losses_comb, losses_comb_s.item())
                self.steps_list = np.append(self.steps_list, num_steps)

                # Reset the cumulative step loss
                losses_comb_s *= 0


            # Update EMA on CPU every `update_frequency` batches
            if num_steps % self.ema_update_freq == 0:
                with torch.no_grad():
                    for ema_param, param in zip(self.ema_model_cpu.parameters(), self.model.module.parameters()):
                        if param.requires_grad:
                            ema_param.data.mul_(self.ema_decay).add_(param.cpu().data, alpha=(1.0 - self.ema_decay))


            # Save the EMA model and graph every number of desired steps
            if num_steps%self.numSaveSteps == 0 and is_main_process(self.subgroup):
                self.ema_model_cpu.wandb_id = self.wandb_id
                self.ema_model_cpu.saveModel(self.saveDir, self.optim, self.scheduler, self.grad_scaler, num_steps)
                # self.graph_losses()

                print("Saving model")
        
        # if is_main_process():
        #     print(f"Loss at step #{num_steps}, update #{num_steps/self.numSteps}\n"+\
        #             f"Combined: {round(self.losses_comb[-10:].mean(), 4)}\n\n")


