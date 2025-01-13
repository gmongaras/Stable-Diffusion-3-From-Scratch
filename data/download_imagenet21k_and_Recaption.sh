#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p batch
#SBATCH -o runjob_download_imagenet21k_and_Recaption.out
#SBATCH --mem=500G
#SBATCH --gres=gpu:1

export HF_HOME=/users/gmongaras/gmongaras_diffusion_models/Stable_Diffusion_3/data/cache

cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/
git clone https://huggingface.co/datasets/gmongaras/Stable_Diffusion_3_Recaption
rm -rf Stable_Diffusion_3_Recaption/.git
git clone https://huggingface.co/datasets/gmongaras/Imagenet21K
rm -rf Imagenet21K/.git


cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3
python - << EOF
import datasets
dataset = datasets.load_dataset("parquet", data_files=f"data/Stable_Diffusion_3_Recaption/data/*.parquet", cache_dir="data/cache", split="train")
EOF

python - << EOF
import datasets
dataset = datasets.load_dataset("parquet", data_files=f"data/Imagenet21K/data/*.parquet", cache_dir="data/cache", split="train")
EOF
