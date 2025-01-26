#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p batch
#SBATCH -o recaption_parquets.out.1
#SBATCH --mem=750G
#SBATCH --gres=gpu:1



export CUDA_VISIBLE_DEVICES="0"
BATCH_NUM=1
GPU_NUM=0

cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/
python data/recaption_parquets.py $BATCH_NUM $GPU_NUM
