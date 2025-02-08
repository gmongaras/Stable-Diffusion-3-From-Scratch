#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p batch
#SBATCH --gres gpu:1
#SBATCH -o upload_imagenet_2021.out
#SBATCH --mem=500G

cd /users/gmongaras/gmongaras_diffusion_models/Stable_Diffusion_3/
python data/upload_imagenet2021_and_CC12M.py