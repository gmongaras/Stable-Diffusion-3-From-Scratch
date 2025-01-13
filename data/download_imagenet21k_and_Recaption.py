from datasets import load_dataset
from huggingface_hub import snapshot_download

snapshot_download("gmongaras/Imagenet21K", repo_type="dataset", cache_dir="data/cache")
snapshot_download("gmongaras/Stable_Diffusion_3_Recaption", repo_type="dataset", cache_dir="data/cache")

# try:
#     ds = load_dataset("gmongaras/Imagenet21K", data_dir="data/Imagenet21K_", cache_dir="data/cache", num_proc=8)
# except:
#     ds = load_dataset("gmongaras/Imagenet21K", data_dir="data/Imagenet21K_", cache_dir="data/cache", num_proc=8, download_mode="force_redownload")
# ds2 = load_dataset("gmongaras/Stable_Diffusion_3_Recaption", cache_dir="data/cache", num_proc=8)
# pass