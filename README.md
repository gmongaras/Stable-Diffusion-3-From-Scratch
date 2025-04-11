# Summary
This is work in progress. Will update the README soon!

# Checkpoints

A list of checkpoints can be found at [https://huggingface.co/collections/gmongaras/stable-diffusion-3-checkpoints-67f91e538138a2960a81eeb7](https://huggingface.co/collections/gmongaras/stable-diffusion-3-checkpoints-67f91e538138a2960a81eeb7)

A decent model for V1 models is checkpoint 330000 with [this pkl](https://huggingface.co/gmongaras/datav3_attempt4_8GPU_SoftFlash_RoPE2dV2_2AccSteps_stage2/blob/main/model_330000s.pkl) and [this json](https://huggingface.co/gmongaras/datav3_attempt4_8GPU_SoftFlash_RoPE2dV2_2AccSteps_stage2/blob/main/model_params_330000s.json) which was finetuned on a resolution of 512x512. For some reason the quality decreased as I kept finetuning on 512x512 and I'm still trying to figure out why.

To do inference, take a look at the `src/infer_loop.ipynb` file. Note that you only need to download the `model_ema_....pkl` and `model_params_....json` files for inference. Training requires the other files.


# Finetuning for higher resolution

1. To finetune on higher resolution, set `loadModel` to True and set the load paths to the checkpoint of the checkpoint you want to finetune. 
2. Set `wandb_name` and `saveDir` to the new name if you want to use a new name for the finetuning procedure.
3. Change `reset_wandb` and `reset_optim` to True. This will reset the wandb id to start a new run and reset the optimizer states since they are no longer valid.
4. Keeping `max_res_orig` the same, change `max_res` to the new resolution (this changes how the data and model gpus communicate)
5. Change `bucket_indices_path` and `data_parquet_folder` to the paths of the new dataset
6. You probably want to decrease the batch size as well