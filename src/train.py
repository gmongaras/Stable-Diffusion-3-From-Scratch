import pickle
from models.diff_model import diff_model
from model_trainer import model_trainer
import os
import click
from typing import List





def train():
    totalSteps = 250_000
    batchSize = 128
    inCh = 4
    num_classes = 1000
    patch_size = 2
    num_blocks = 20
    dim = int(64*num_blocks)
    c_dim = 512
    hidden_scale = 4.0
    num_heads = num_blocks
    attn_type = "cosine"
    device = "gpu"
    wandb_name = "del"
    log_steps = 10
    p_uncond = 0.1
    lr = 1e-4
    use_lr_scheduler = False
    ema_update_freq = 100
    ema_decay = 0.99
    warmup_steps = 1000

    numSaveSteps = 1000
    saveDir = "models/del"

    loadModel = False
    loadDir = "models/del"
    loadFile = "model_68000s.pkl"
    loadDefFile = "model_params_68000s.json"
    optimFile = "optim_68000s.pkl"
    schedulerFile = "scheduler_68000s.pkl"
    scalerFile = "scaler_68000s.pkl"
    
    
    
    ### Model Creation
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
        p_uncond=p_uncond,
        optimFile=None if loadModel==False or optimFile==None else loadDir+os.sep+optimFile,
        schedulerFile=None if loadModel==False or schedulerFile==None else loadDir+os.sep+schedulerFile,
        scalerFile=None if loadModel==False or scalerFile==None else loadDir+os.sep+scalerFile,
        use_amp=True,
        wandb_name=wandb_name,
        log_steps=log_steps,
        device=device,
    )
    trainer.train()
    
    
    
    
    
if __name__ == '__main__':
    train()
