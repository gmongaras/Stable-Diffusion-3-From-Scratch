#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p standard-s
#SBATCH -o runjob_cc12m.out
#SBATCH --mem=500G

cd ~/gmongaras_diffusion_models/Stable_Diffusion_3/data/
bash download_cc12m.sh
cd ../
bash data/merge_cc12m.sh