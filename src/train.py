import pickle
from models.diff_model import diff_model
from model_trainer import model_trainer
import os
import click
from typing import List





def train():
    totalSteps = 250_000
    batchSize = 96
    inCh = 4
    num_loader_gpus = 2
    num_model_gpus_per_loader = 3 # Total GPU count = num_loader_gpus + num_loader_gpus * num_model_gpus_per_loader
    num_gpus_per_device = 8 # Number of GPUs per device such as 8 for A100
    # class_dim = 1792
    class_dim = 2048
    patch_size = 2
    num_blocks = 19
    dim = int(64*num_blocks)
    hidden_scale = 4.0
    num_heads = num_blocks
    attn_type = "cosine"
    device = "gpu"
    wandb_name = "attempt1"
    log_steps = 10
    null_prob_L4 = 0.464
    null_prob_G14 = 0.464
    null_prob_T5 = 0.464
    lr = 1e-4
    use_lr_scheduler = False
    ema_update_freq = 100
    ema_decay = 0.99
    warmup_steps = 1000
    checkpoint_MLP = True

    numSaveSteps = 1000
    saveDir = "models/attempt1"

    loadModel = False
    loadDir = "models/attempt1"
    loadFile = "model_68000s.pkl"
    loadDefFile = "model_params_68000s.json"
    optimFile = "optim_68000s.pkl"
    schedulerFile = "scheduler_68000s.pkl"
    scalerFile = "scaler_68000s.pkl"
    
    
    
    ### Model Creation
    model = diff_model(
        inCh=inCh,
        class_dim=class_dim,
        patch_size=patch_size,
        dim=dim,
        hidden_scale=hidden_scale,
        num_heads=num_heads,
        attn_type=attn_type,
        num_blocks=num_blocks,
        checkpoint_MLP=checkpoint_MLP,
        device=device,
    )
    
    # Optional model loading
    if loadModel == True:
        model.loadModel(loadDir, loadFile, loadDefFile)
    
    # Train the model
    trainer = model_trainer(
        diff_model=model,
        batchSize=batchSize, 
        numSteps=1,
        totalSteps=totalSteps, 
        lr=lr, 
        ema_update_freq=ema_update_freq,
        ema_decay=ema_decay,
        warmup_steps=warmup_steps,
        use_lr_scheduler=use_lr_scheduler,
        saveDir=saveDir,
        numSaveSteps=numSaveSteps,
        null_prob_L4=null_prob_L4,
        null_prob_G14=null_prob_G14,
        null_prob_T5=null_prob_T5,
        optimFile=None if loadModel==False or optimFile==None else loadDir+os.sep+optimFile,
        schedulerFile=None if loadModel==False or schedulerFile==None else loadDir+os.sep+schedulerFile,
        scalerFile=None if loadModel==False or scalerFile==None else loadDir+os.sep+scalerFile,
        use_amp=True,
        wandb_name=wandb_name,
        log_steps=log_steps,
        device=device,
        num_loader_gpus=num_loader_gpus,
        num_model_gpus_per_loader=num_model_gpus_per_loader,
        num_gpus_per_device=num_gpus_per_device
    )
    trainer.train()
    
    
    
    
    
if __name__ == '__main__':
    train()
