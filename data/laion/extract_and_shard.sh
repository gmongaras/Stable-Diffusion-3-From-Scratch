#!/bin/bash
#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=extract_and_shard
#SBATCH --output=extract_and_shard.out
#SBATCH --partition=standard-s
#SBATCH --nodes=1                 # Update with the default number of nodes
#SBATCH --ntasks-per-node=1        # One task per node
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --mem=500G                  # Memory per node

cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/
# source ~/.bashrc

# git clone https://gmongaras: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@huggingface.co/datasets/laion/relaion2B-en-research
# git clone https://gmongaras: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@huggingface.co/datasets/laion/relaion2B-multi-research
# git clone https://gmongaras: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@huggingface.co/datasets/laion/relaion2B-nolang-research

python data/laion/extract_and_shard.py
