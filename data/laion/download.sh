#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p standard-s
#SBATCH -o download.out
#SBATCH --mem=100G
##SBATCH --gres=gpu:1



### NOTE: If you want to checkpoint, just put a bunch of xxxxxx_stats.json files in that directory starting from
###       000000_stats.json and then run this script. It will pick up from the last .json it sees.


cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/
python data/laion/download.py
