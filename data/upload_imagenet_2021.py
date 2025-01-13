import datasets
from datasets import load_dataset

# Replace 'path/to/folder' with the path to your folder containing the Parquet files
parquet_folder = "data/Imagenet21"

# Load the dataset from the folder of Parquet files
dataset = load_dataset("parquet", data_files=f"{parquet_folder}/*.parquet", cache_dir="data/cache")

# Check the dataset structure
print(dataset)
print("Data loaded. Saving to hub")

# Shuffle the data
# Shuffling makes uploading insanely slow
# dataset = dataset.shuffle(seed=42)

# Get the token from the .env file
with open(".env", "r") as f:
    token = f.read().strip()

# # Iterate over all shards
# num_shards = 7760
# for i in range(num_shards):
#     # Shard the dataset
#     shard = dataset["train"].shard(num_shards=num_shards, index=i)

#     # Upload the shard to the hub
#     print(f"Pushing shard {i} to hub")
#     shard._push_parquet_shards_to_hub("gmongaras/Imagenet21", token=token, num_shards=1)
#     print(f"Shard {i} pushed to hub")

# Use this to get the number of optimal shards
dataset.push_to_hub("gmongaras/Imagenet_2021", token=token)