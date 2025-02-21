import os
os.environ["HF_HOME"] = "data/cache"
import datasets
from datasets import load_dataset

# Replace 'path/to/folder' with the path to your folder containing the Parquet files
parquet_folder = "data/cc12m_and_imagenet21K_highqual"
dataset_name = "gmongaras/CC12M_and_Imagenet21K_Recap_Highqual"

# Load the dataset from the folder of Parquet files
dataset = load_dataset("parquet", data_files=f"{parquet_folder}/*.parquet", cache_dir="data/cache", split="train", num_proc=64)
### NOTE: If the above runs int oa weird index error, I added to the file "/python_pacakages/datasets/packaged_modules/parquet/parquet.py"
###       after line 102 the following lines:
###       if "__index_level_0__" in pa_table.column_names:
###           pa_table = pa_table.drop(['__index_level_0__'])
### Which will remove the index column from the table if it exists



# Check the dataset structure
print(dataset)
print("Data loaded. Saving to hub")

# Get the token from the .env file
with open(".env", "r") as f:
    token = f.read().strip()

#"""
# Iterate over all shards
num_shards_start = 7020 # Number uploaded plus 1
num_shards = 8477 # Get this by running the function below
num_shards_per_push = 15 # You want this relatively high so you don't run into rate limits but not too high so it can't push the shards as it will timeout

num_shards = (num_shards // num_shards_per_push) + 1
num_shards_start = num_shards_start // num_shards_per_push
for i in range(num_shards_start, num_shards):
    # Shard the dataset
    shard = dataset.shard(num_shards=num_shards, index=i)

    # Upload the shard to the hub
    print(f"Pushing shard {i} ({i*num_shards_per_push} to {(i+1)*num_shards_per_push}) to hub")
    ### NOTE: To get this to work, you need to change our your
    ###       `site-packages/datasets/arrow_dataset.py` file with the one in this folder __arrow_dataset.py
    ###       and change the `site-packages/datasets/dataset_dict.py` file with the one in this folder __dataset_dict.py
    ###       I am using datasets version 2.16.0
    try:
        shard.push_to_hub(dataset_name, token=token, start__=i*num_shards_per_push, num_shards=num_shards_per_push)
    except:
        shard.push_to_hub(dataset_name, token=token, start__=i*num_shards_per_push, num_shards=num_shards_per_push)
    print(f"Shard {i} pushed to hub")
#"""

# # Use this to get the number of optimal shards
# dataset.push_to_hub(dataset_name, token=token)