#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Mrrrowww_^w^
#SBATCH -p standard-s
#SBATCH -o runjob.out
#SBATCH --mem=500G

# # If it doesn't exist, get the tsv file
# if [ ! -f cc12m.tsv ]; then
#     wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv
#     sed -i '1s/^/url\tcaption\n/' cc12m.tsv
# fi

# # # Downlaod dataset
# # img2dataset --url_list cc12m.tsv --input_format "tsv"\
# #          --url_col "url" --caption_col "caption" --output_format webdataset\
# #            --output_folder cc12m --processes_count 16 --thread_count 64 --image_size 256\
# #              --enable_wandb False --resize_model "center_crop" --retries 5 \
# #              --incremental "incremental"
# # no resize
# img2dataset --url_list cc12m.tsv --input_format "tsv"\
#          --url_col "url" --caption_col "caption" --output_format webdataset\
#            --output_folder cc12m --processes_count 16 --thread_count 64 --image_size 10000000\
#              --enable_wandb False --resize_model "no" --retries 5 \
#              --incremental "incremental"




# https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC12M
git clone https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC12M
mv LLaVA-ReCap-CC12M cc12m

# https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext
git clone https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext
mv conceptual-captions-cc12m-llavanext cc12m_captions
gunzip cc12m_captions/train.jsonl.gz cc12m_captions/