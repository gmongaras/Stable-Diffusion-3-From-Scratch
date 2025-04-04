from PIL import PngImagePlugin
import src.helpers.dataset_utils as dataset_utils
import datasets

# Needed to prevent error with large text chunks - I just set it to a shit ton
PngImagePlugin.MAX_TEXT_CHUNK = 1000000 * 1024 * 1024

bucket_indices_path = "data/bucket_indices_512.npy"
data_parquet_folder = "data/cc12m_and_imagenet21K_highqual_512"
n_proc = 8

dataset = datasets.load_dataset("parquet", data_files=f"{data_parquet_folder}/*.parquet", cache_dir="data/cache", split="train", num_proc=64)
dataset_utils.load_indices(bucket_indices_path, dataset, n_proc)