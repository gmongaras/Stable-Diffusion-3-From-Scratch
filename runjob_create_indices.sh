#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=^w^_UwU_>w<
#SBATCH -p highmem
#SBATCH -o runjob_create_indices.out
#SBATCH --mem=500G
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1


cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/
python src/create_indices.py
