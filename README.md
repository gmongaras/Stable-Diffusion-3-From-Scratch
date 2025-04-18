# Summary

I wanted to try to train Stable Diffusion 3 from scratch, but with 8 GPUs. Did I? That's probably up to debate. However an attempt was made. The images look mid as I only have 8 GPUs and finite time but the scripts could be useful to others perhaps.

# Contents
- [Setup](#setup)
- [Downloading Pretrained Models](#downloading-pretrained-models)
- [Inference](#inference)
- [Training](#Training)
- [Model Details](#model-details)
- [Recpationing and Other Scripts](#recpationing-and-other-scripts)











# Setup

First clone the repo, and create a venv.

```
git clone https://github.com/gmongaras/Stable-Diffusion-3-From-Scratch.git ./Stable_Diffusion_3
cd Stable_Diffusion_3
python3.10 -m venv SD3Venv
source SD3Venv/bin/activate
pip install pip -U
```

Install the version of torch that fits your GPU from `https://pytorch.org/get-started/locally/`. I used `torch 2.6.0, cu118` and `torchvision 0.21.0, cu118`. Any version of Cuda should work as long as it's supported by torch, but I would stick with torch 2.6.0 or torchvision 0.21.0 in case there are deprecation issues. Below is an example of installing a specific version for cuda 11.8:
```
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Also install the verison of xformers that fits your gpu from `https://github.com/facebookresearch/xformers`. Again, any version of cuda should work. As for the package verison I used `version 0.0.29.post3`. This was the latest at the time.

I use [flash attention](https://github.com/Dao-AILab/flash-attention) in this repo. You can install it with the following. Note that I am putting the version just in case there are deprecation issues.
```
pip install wheel
pip install flash-attn==2.6.3 --no-build-isolation
```

Install the rest of the requirements with
```
pip install -r requirements.txt
```

Most specific versions probably don't matter except for the transformers version as the transformers package changes a lot between versions.













# Downloading Pretrained Models

A list of checkpoints can be found at [https://huggingface.co/collections/gmongaras/stable-diffusion-3-checkpoints-67f91e538138a2960a81eeb7](https://huggingface.co/collections/gmongaras/stable-diffusion-3-checkpoints-67f91e538138a2960a81eeb7)

Currently the best model I have trained is located in [this repo](https://huggingface.co/gmongaras/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_40batchsize_stage2). The highest version is likely the best model I have trained, but I'll leave a note if that's not the case. Currently, the best model is checkpoint 475,000. Note liks will mention this version though there is likely a better version of the model which you might want to download instead. For inference with checkpoint 475,000, you want to download [this pickle](https://huggingface.co/gmongaras/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_40batchsize_stage2/blob/main/model_ema_475000s.pkl) and [this json](https://huggingface.co/gmongaras/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_40batchsize_stage2/blob/main/model_params_475000s.json) and put it somewhere you can reference in the inference script. If you want to finetune this model, you probably want [the optimizer states](https://huggingface.co/gmongaras/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_40batchsize_stage2/blob/main/optim_475000s.pkl), [the fast moving model](https://huggingface.co/gmongaras/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_40batchsize_stage2/blob/main/model_475000s.pkl), [the scalar state](https://huggingface.co/gmongaras/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_40batchsize_stage2/blob/main/scaler_475000s.pkl), and [the scheduler state](https://huggingface.co/gmongaras/datav3_attempt5_8GPU_SoftFlash_RoPE2d_2AccSteps_40batchsize_stage2/blob/main/scheduler_475000s.pkl). This checkpoint is about 1.2B params and was finetuned on resolution 512x512 after taining for a resolution of 256x256 for 300,000 steps. This means the max resolution is 512x512. I also trained on almost all multiples of 16. So any multiple of 16 such as 256x128 should work, allowing for different aspect ratio outputs.
















# Inference

The main inference script is `src/infer_loop.ipynb` as it allows you to load a model once and sample multiple times. However I also have a python inference script at `infer.py`. I update the python script a lot less, though it's easier to debug and it outputs a gif of the diffusion process.

Before running inference, make sure you [downloaded the pretrained model(s)](#Downloading-Pretrained-Models) you want to do inference with. Note that for inference you only need the `model_ema_....pkl` file and the `model_params_....json` file.

Running the script is easy, just run it like you would any other ipynb script. It has the following params that can be changed:
- loadDir - Directory to load the model from
- loadFile - File to load the model from
- loadDefFile - File to load the model hyperparams from
- device - Device to put the model on
- batch_size - Number of images to produce in parallell
- num_steps - Number of steps to take when generating the image
- text_input - Text prompt for the image
- width - Width of the image. Note that this must be a factor of 16 for all my pretrained models.
- height - Height of the image. Note that this must be a factor of 16 for all my pretrained models.
- CFG_scale - CFG value. The higher, the more the model follows the prompt. A value around 5-7 works well.
- sampler - Sampler to use. I have a couple I was experimenting with (`euler`, `euler_stochastic`, `heun`)
- seed - Keep at -1 to have a random seed. Change to seed the randomness.

The script takes about 18GB of VRAM to run at 512x512 resolution. It was not optimized at all lol


















# Training


The training script is lcoated in `src/train.py`. While you could go and just run the script immediately, it will likely timeout as one of the data loaders has to download and the nindex the entire dataset. Instead you should first run `src/create_indices.py` then `src/train.py`.

Unlike an LLM, diffusion models have a lot of overhead preprocessing the data. While the original stable diffusion 3 paper took the entire dataset and preprocessed all the data by tokenizing the text and processing the images through the VAE, I wanted to experiment with the data. The nice part about the SD3 approach is it is fast to load in and makes everything easy. The downside is it takes forever to tokenize all your data and you cannot change the data once it's processed (unless you reprocess all you data).

Instead, I decided to use some GPUs to do forward/backward passes on the model and have other GPUs preprocess and load in new data. For every X "model GPUs" that do a forward/backward pass, a single "data loader" GPU can load data for all X GPUs. In my case I had X=3 model GPUs. Idealy the time to processes the data should be equal to a single forward/backward pass, thus having very little overhead between forward passes.

The current configuration works well for 8 A100 GPUs. It should support more nodes, but I haven't tested it too well.

If you are planning to train from scratch, I would



## Create a subset

Before training, you need to bucket you data. The base dataset is located at [https://huggingface.co/datasets/gmongaras/CC12M_and_Imagenet21K_Recap_Highqual](https://huggingface.co/datasets/gmongaras/CC12M_and_Imagenet21K_Recap_Highqual). This is a dataset of images with their corresponding text captions, image size, and aspect ratio at full resolution. Download this dataset either by using huggingface or by cloning the repo into a directory.

`data/create_phase.py` takes the input directory above `input_dir`, a `max_resolution`, a `patch_size`, and and output directory. All images are resized to the nearest factor of `patch_size`, retaining their aspect ratio. A new dataset with bucket classes and the resized images is created.



## Creating Indices

After either downloading the subset dataset or making one, you want to create indices using the `src/create_indices.py` file. This takes `bucket_indices_path`, `data_parquet_folder`, and `n_proc` as params. The script takes the folder of parquet files `data_parquet_folder` which is the subset from above and outputs a dictionary or bucket index mapping in `bucket_indices_path`. `n_proc` controls how many processes to do this in parallel.

This script can take about a day to complete.





## Training the model

I trained this model on 8 A100 GPUs on a single node so all my configurations are tuned for this setup. The train script should be straightforward to run with the SLURM script in `runjob_SLURM.sh` if you change `nnodes` to the number of nodes in your cluster and `nproc_per_node` to the number of GPUs per node. 

The train script has the following parameters:
- totalSteps - Number of steps to step the optimizer to update the model
- batchSize - Per-GPU batch size
- accumulation_steps - Number of steps to accumulate gradients before updating the model. Note that this increases the virtual batch size by this factor. This is just a way of increasing batch size by doing more forward passes if memory constrained. So, the model gradients see a batch size of batchSize*accumulation_steps while your GPUs see a batch size of batchSize.
- inCh - Number of input channels into the model. Dont change this unless changing the VAE.
- loader_to_model_gpu - This maps ids for each "data loader" GPU to ids for each "model GPU". The keys are the data loader GPUs and values are lists of model GPUs that data loader is responsible for. For example with 3 GPUs, I can have a map `{ 0: [1, 2] }` which states that GPU 0 is a data loader GPU and will be responsible for processing the data for GPUs 1 and 2, which will do passes and updates to the model. I optimzied this code such that each data gpu gets three model gpus max. The more data GPUs you have, the less your model updates. The more model GPUs you have, the more overhead between processing data and updating the model. Currently, the is very little overhead for the 1-3 relationship where one data gpu gets three model gpus. The default in my code is probably what should be used unless you are using fewer than 8 GPUs.
- class_dim - Dimension of the input class vector. That is, the dimension of the CLIP pooled output. This shouldn't be changed unless you ar echanging the text models. 
- patch_size - Size of each patch in the input image. A patch size of 2 means the input images is split into 2x2 nonoverlapping patches where each patch is a token.
- num_blocks - Number of transformer blocks in the model
- dim - Dimension the transformer operates on. The current formula is straight from the SD3 paper.
- hidden_scale - How much to scale the dim for the intermediate output in the MLP layers.
- num_heads - Number of heads in the attention layers
- attn_type - Attention type to use. I would just stick with `softmax_flash`, but I experimented with others (`softmax`, `cosine`)
- MLP_type - Either use a SwiGLU MLP or GeLU MLP (`gelu` or `swiglu`). Note that SwiGLU adds another linear layer over gelu.
- device - Just keep this `gpu`
- wandb_name - Name of the wandb project for training curves
- wandb_log_gradients - Flag to turn off/on wandb gradient logging. I found that this increases memory usage and turned it off. Could be helpful for debugging though
- log_steps - Number of steps to take until logging to wandb
- bucket_indices_path - Path to the bucket indices created above
- data_parquet_folder - Path to the parquet folder the bucket indices were created for
- max_res - Max resolution for the model. That is, the max bucket size. (useful for finetuning RoPE)
- max_res_orig - Max resolution the model was originally trained on. That is, the max bucket size. (useful for finetuning RoPE)
- null_prob_pooled - Probability of dropping the pooled embedding. Note that there is only one pooled model output so I have this set to 0.1 or 10%.
- null_prob_gemma/null_prob_bert - Probability of dropping the embedding of either of these models. Note that I have this set to 0.316 for each of these models. This is because if we take the multiplicative probability of dropping both 0.316*0.316, this is 0.1 or 10%. Thus we drop the entire embedding 10% of the time. This is the same idea used in the SD3 paper except they had three embeddings while I have two.
- lr - Learning rate for the model
- use_lr_scheduler - True to have a decay in learning rate over time, False otherwise.
- ema_update_freq - How many steps to take before updating the EMA model.
- ema_decay - Percept of the EMA model to keep when updating it. The lower, the faster it updates.
- warmup_steps - Number of steps before hitting the learning rate. The warmup period goes from 0 to lr in this number of steps.
- checkpoint_MLP - True to checkpoint MLPs to convserve memory, False to not checkpoint.
- checkpoint_attn - True to checkpoint attention layers to convserve memory, False to not checkpoint.
- positional_encoding - The positional encoding to use. SD3 used absolute. I used RoPE2d and had a couple experiments with NoPE as well as a positional encoding I was developing (`absolute` or `RoPE` or `NoPE` or `RoPE2d` or `RoPE2dV2`)
- kv_merge_attn - Expierment I was running where keys and values were merged before the attention layer
- qk_half_dim - Experiment I was running where the query and key dimension was halved.
- text_loss_weight - Expierment I was running where the output of the text embeddings had an NSP loss. Idea was that the model would be forced to learn how to model language. I didn't play with this much. A value of 0 means no text loss. A value greater than 0 means use text loss.
- reset_wandb - True to reset wandb upon reloading a checkpoint.
- reset_optim - True to reset optimizer states upon reloading a checkpoint.
- numSaveSteps - Number of steps before saving the model
- saveDir - Directory to save the model to. Note tht models are saved step-wise so they don't overwrite each other.
- loadModel - True to load a checkpoint, False to not.
- loadDir/loadFile/load_ema_file/loadDefFile/optimFile/schedulerFile/scalerFile - Directory and file names to load a checkpoint from if loading a checkpoint

If all goes well, the thing should train!



## Finetuning for higher resolution

1. To finetune on higher resolution, set `loadModel` to True and set the load paths to the checkpoint of the checkpoint you want to finetune. 
2. Set `wandb_name` and `saveDir` to the new name if you want to use a new name for the finetuning procedure.
3. Change `reset_wandb` and `reset_optim` to True. This will reset the wandb id to start a new run and reset the optimizer states since they are no longer valid.
4. Keeping `max_res_orig` the same, change `max_res` to the new resolution (this changes how the data and model gpus communicate)
5. Change `bucket_indices_path` and `data_parquet_folder` to the paths of the new dataset
6. You probably want to decrease the batch size as well















# Model Details

The model mostly follows the original stable diffusion 3 architecture, but below are a list of changes I remember making:
1. Instead of using the two CLIP models and T5, I used ModernBert, Gemma, and Meta CLIP. ModernBERT is fast. Gemma was used by an Nvidia paper and they said it was pretty good (it's also faster than T5). Meta CLIP is just CLIP but newer. I used Meta CLIP solely for the pooled embedding and I used the other two for the large embedding going into the transformer
2. I use 2d RoPE instead of 2d absolute positional encodings
3. I normalize each embedding seperately as the staistics for each are different. I also add a learnable scalar as some of the outputs of the models are massive in magnitude.
4. I have a learnable time scale such that the model can learn the range it wants to put the timestep betwee. I start this off at 0-1000, but it can change it.
5. I fix the skip connection in each transformer block. The diagram shows a skip connection from the input into the block to the output of attention and output of the MLP. This should be a skip connection from the block input to the attention output, then a skip connection from the attention output to the MLP output like in a normal transformer.
6. Instead of applying SiLU to the timestep embedding every layer (which makes no sense. Just do it once), I apply an MLP then SiLU.
7. I actually recaption all images





















# Recpationing and Other Scripts

I have a ton of scripts in `data` and really don't feel like describing each. I will just describe the important ones here.

## Laion download

A lof of script wrt. downloading Laion can be found in `data/laion`. If you want to download Laion, I would use these as it is very very annoying to setup. I also shard the data to parquets in `data/laion/extract_and_shard.py` so the data is actually usable.

I just didn't have enough space to do anything with this data, but whatever. Hopefully someone else out there finds these scripts useful.


## Recpationing

Instead of just taking the data and working with the raw captions as Stable Diffusion 3 did, it's much much better to recaption. My recaption script can be found in `data/recaption_parquets.py`. This script essentially takes a folder of parquets and recaptions them using llava and llama to make the captions shorter. Thank you to [Caption Emporium](https://huggingface.co/CaptionEmporium) for describing their method. I basically just used that and it worked quite well. As for how it works it's basically just prompting the model and sitting around until it captions everything.

