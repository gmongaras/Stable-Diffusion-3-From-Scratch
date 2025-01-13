import datasets
from datasets import load_dataset

# Replace 'path/to/folder' with the path to your folder containing the Parquet files
parquet_folder = "data/final_dataset"

# Load the dataset from the folder of Parquet files
dataset = load_dataset("parquet", data_files=f"{parquet_folder}/*.parquet", cache_dir="data/cache")

# Check the dataset structure
print(dataset)
print("Data loaded. Saving to hub")

# Upload the dataset to the Hugging Face Hub
with open(".env", "r") as f:
    token = f.read().strip()
dataset.push_to_hub("gmongaras/Stable_Diffusion_3_Recaption", token=token)

print("Saved to hub. Loading into dataset folder")
load_dataset("gmongaras/Stable_Diffusion_3_Recaption", cache_dir="data/Stable_Diffusion_3_Recaption")