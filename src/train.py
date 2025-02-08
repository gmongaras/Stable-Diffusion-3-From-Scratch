import pickle
from models.diff_model import diff_model
from model_trainer import model_trainer
import os
import click
from typing import List





def train():
    totalSteps = 250_000
    batchSize = 120
    inCh = 16
    # num_loader_gpus = 2
    # num_model_gpus_per_loader = 3 # Total GPU count = num_loader_gpus + num_loader_gpus * num_model_gpus_per_loader
    # num_gpus_per_device = 8 # Number of GPUs per device such as 8 for A100
    loader_to_model_gpu = {
            0: [2, 3, 4],
            1: [5, 6, 7],
            8: [10, 11, 12],
            9: [13, 14, 15],
            16: [18, 19, 20],
            17: [21, 22, 23],
            24: [26, 27, 28],
            25: [29, 30, 31],
        }
    # loader_to_model_gpu = {
    #        0: [1],
    #    }
    # class_dim = 1792
    class_dim = 768
    patch_size = 2
    num_blocks = 19
    dim = int(64*num_blocks)
    hidden_scale = 4.0
    num_heads = num_blocks
    attn_type = "cosine"
    device = "gpu"
    # wandb_name = "attempt4_16GPU_RoPE_Cos_Clip_Fixmag_Merge"
    wandb_name = "datav2_attempt2_8GPU_Cos_Redo"
    log_steps = 10
    null_prob_pooled = 0.1
    null_prob_gemma = 0.1
    null_prob_bert = 0.1
    lr = 1e-4
    use_lr_scheduler = False
    ema_update_freq = 100
    ema_decay = 0.99
    warmup_steps = 1000
    checkpoint_MLP = True
    positional_encoding = "RoPE" # "absolute" or "RoPE" or "NoPE"
    kv_merge_attn = False
    qk_half_dim = False

    numSaveSteps = 1000
    saveDir = "models/datav2_attempt2_8GPU_Cos_Redo"

    loadModel = True
    loadDir = "models/datav2_attempt2_8GPU_Cos_Redo"
    loadFile = "model_50000s.pkl"
    load_ema_file = "model_ema_50000s.pkl"
    loadDefFile = "model_params_50000s.json"
    optimFile = "optim_50000s.pkl"
    schedulerFile = "scheduler_50000s.pkl"
    scalerFile = "scaler_50000s.pkl"
    
    
    
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
        positional_encoding=positional_encoding,
        kv_merge_attn=kv_merge_attn,
        qk_half_dim=qk_half_dim,
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
        null_prob_pooled=null_prob_pooled,
        null_prob_gemma=null_prob_gemma,
        null_prob_bert=null_prob_bert,
        load_ema_file=None if loadModel==False or load_ema_file==None else loadDir+os.sep+load_ema_file,
        optimFile=None if loadModel==False or optimFile==None else loadDir+os.sep+optimFile,
        schedulerFile=None if loadModel==False or schedulerFile==None else loadDir+os.sep+schedulerFile,
        scalerFile=None if loadModel==False or scalerFile==None else loadDir+os.sep+scalerFile,
        use_amp=True,
        wandb_name=wandb_name,
        log_steps=log_steps,
        device=device,
        loader_to_model_gpu = loader_to_model_gpu
    )
    trainer.train()
    
    
    
    
    
if __name__ == '__main__':
    train()
