#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p batch
#SBATCH -o runjob_download_imagenet_2021.out
#SBATCH --mem=500G
#SBATCH --gres=gpu:1

cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data
wget https://www.image-net.org/data/winter21_whole.tar.gz
tar -xvzf winter21_whole.tar.gz
mv winter21_whole Imagenet21


cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/
python data/convert_imagenet_parquet.py
python data/upload_imagenet_2021.py