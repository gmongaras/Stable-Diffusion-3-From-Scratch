#!/bin/bash
#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=laion_download
#SBATCH --output=laion_download.log
#SBATCH --error=laion_download.err
#SBATCH --partition=standard-s
#SBATCH --nodes=5                 # Update with the default number of nodes
#SBATCH --ntasks-per-node=1        # One task per node
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --mem=250G                  # Memory per node
#SBATCH --exclusive

# cd /projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/laion
# source ~/.bashrc

# git clone https://gmongaras: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@huggingface.co/datasets/laion/relaion2B-en-research
# git clone https://gmongaras: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@huggingface.co/datasets/laion/relaion2B-multi-research
# git clone https://gmongaras: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@huggingface.co/datasets/laion/relaion2B-nolang-research

### NOTE: If the master fails to bind, edit the below so the host
###       is set to an open IP

srun bash download2_.sh
