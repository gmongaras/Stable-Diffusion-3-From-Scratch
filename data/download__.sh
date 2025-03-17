#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=^w^_UwU_>w<
#SBATCH -p standard-s
#SBATCH -o create_phase.out
#SBATCH --mem=500G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64



cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data
git clone https://huggingface.co/datasets/gmongaras/CC12M_and_Imagenet21K_Recap 