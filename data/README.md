1. download_caption_imagenet.sh - Captions ImageNet and moves it to a new folder of .parquet files (final_dataset)
2. download_cc12m.sh - Downloads all of the cc12m dataset and captions for it
3. merge_cc12m.sh - Merges the cc12m dataset with the captions and puts the output in final_dataset
4. make_final.sh - Takes the final_dataset folder and uploads it to the huggingface hub
5. download_imagenet_2021.sh - Downloads the 2021 ImageNet dataset and saves it to huggingface
6. laion/download2.sh - Downloads all of laion
7. laion/extract_and_shard.sh - Extracts all the outputs of (6) and shards it into another directory, deleting the tar and png along the way while keeping the .json as a checkpoint
8. download_imagenet21k_and_Recaption.sh - Downloads both the Imagenet21K and Stable_Diffusion_3_Recaption.
9. recaption_parquet/recaption_parquets.py - Takes a directory or parquets and recaptions them




datasets:
1. CC12M - https://github.com/google-research-datasets/conceptual-12m
2. ImageNet 2012
3. ImageNet 2021
4. LAION (https://huggingface.co/datasets/laion/relaion2B-en-research, https://huggingface.co/datasets/laion/relaion2B-multi-research, https://huggingface.co/datasets/laion/relaion1B-nolang-research)
5. https://huggingface.co/datasets/activebus/Altogether-FT



Good papers about recaptioning:
1. https://ai.meta.com/research/publications/meta-clip-12/