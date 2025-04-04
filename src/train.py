import pickle
from models.diff_model import diff_model
from model_trainer import model_trainer
import os
import click
from typing import List





def train():
    totalSteps = 350_000
    batchSize = 140 # Stage 1
    # batchSize = 40 # Stage 2
    accumulation_steps = 2
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
    attn_type = "softmax_flash"
    MLP_type = "swiglu" # gelu or swiglu
    device = "gpu"
    # wandb_name = "attempt4_16GPU_RoPE_Cos_Clip_Fixmag_Merge"
    #wandb_name = "datav3_attempt2_8GPU_Cos_RoPE1d_resize256"
    wandb_name = "datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_140batchsize_stage1"
    wandb_log_gradients = False # Turn this off to prevent wandb from hogging gpu memory (illegal memory accesses will occur when memory usage isn't maximum)
    log_steps = 10
    bucket_indices_path = "data/bucket_indices_256.npy"
    data_parquet_folder = "data/cc12m_and_imagenet21K_highqual_256"
    max_res = 256 # Max res in pixel space, not latent space
    max_res_orig = 256
    downsample_factor = 16 # Downsample factor ((downsample scale for VAE) * (patch size))
    # The original paper used 0.464 because 0.464*0.464*0.464 ~= 0.1 for three parts.
    # We want to mask about 10% of the time. Since there's only one pooled, that's just 0.1
    # and since there's two parts to the big embedding, that would be about 0.316
    null_prob_pooled = 0.1
    null_prob_gemma = 0.316
    null_prob_bert = 0.316
    lr = 1e-4
    use_lr_scheduler = False
    ema_update_freq = 100
    ema_decay = 0.99
    warmup_steps = 1000
    checkpoint_MLP = True
    checkpoint_attn = True
    positional_encoding = "RoPE2d" # "absolute" or "RoPE" or "NoPE" or "RoPE2d" or "RoPE2dV2"
    kv_merge_attn = False
    qk_half_dim = False
    text_loss_weight = 0.0 # Zero for no text loss
    reset_wandb = False
    reset_optim = False

    numSaveSteps = 1000
    #saveDir = "models/datav3_attempt2_8GPU_Cos_RoPE1d_resize256"
    saveDir = "models/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_140batchsize_stage1"

    loadModel = True
    #loadDir = "models/datav3_attempt2_8GPU_Cos_RoPE1d_resize256"
    loadDir = "models/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_140batchsize_stage1"
    loadFile = "model_155000s.pkl"
    load_ema_file = "model_ema_155000s.pkl"
    loadDefFile = "model_params_155000s.json"
    optimFile = "optim_155000s.pkl"
    schedulerFile = "scheduler_155000s.pkl"
    scalerFile = "scaler_155000s.pkl"
    
    
    
    ### Model Creation
    model = diff_model(
        inCh=inCh,
        class_dim=class_dim,
        patch_size=patch_size,
        dim=dim,
        hidden_scale=hidden_scale,
        num_heads=num_heads,
        attn_type=attn_type,
        MLP_type=MLP_type,
        num_blocks=num_blocks,
        checkpoint_MLP=checkpoint_MLP,
        checkpoint_attn=checkpoint_attn,
        positional_encoding=positional_encoding,
        max_res_orig=max_res_orig,
        max_res=max_res,
        update_max_res=True,
        kv_merge_attn=kv_merge_attn,
        qk_half_dim=qk_half_dim,
        text_loss=(text_loss_weight > 0.0),
        device=device,
    )
    
    # Optional model loading
    if loadModel == True:
        model.loadModel(loadDir, loadFile, loadDefFile)
    
    # Train the model
    trainer = model_trainer(
        diff_model=model,
        batchSize=batchSize, 
        accumulation_steps=accumulation_steps,
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
        text_loss_weight=text_loss_weight,
        load_ema_file=None if loadModel==False or load_ema_file==None else loadDir+os.sep+load_ema_file,
        optimFile=None if loadModel==False or optimFile==None else loadDir+os.sep+optimFile,
        schedulerFile=None if loadModel==False or schedulerFile==None else loadDir+os.sep+schedulerFile,
        scalerFile=None if loadModel==False or scalerFile==None else loadDir+os.sep+scalerFile,
        use_amp=True,
        wandb_name=wandb_name,
        wandb_log_gradients=wandb_log_gradients,
        reset_wandb=reset_wandb,
        reset_optim=reset_optim,
        log_steps=log_steps,
        device=device,
        loader_to_model_gpu = loader_to_model_gpu,
        bucket_indices_path = bucket_indices_path, 
        data_parquet_folder = data_parquet_folder,
        max_res=max_res,
    )
    trainer.train()
    
    
    
    
    
if __name__ == '__main__':
    train()
