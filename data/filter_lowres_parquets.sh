#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=^w^_UwU_>w<
#SBATCH -p standard-s
#SBATCH -o filter_lowres_parquets.out
#SBATCH --mem=500G



cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/
python data/filter_lowres_parquets.py
