import datasets
import requests
from tqdm import tqdm
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor


# Vars
num_files_per_par = 1000
checkpoint_file = "./data/laion/checkpoint.txt"
failed_file = "./data/laion/failed.txt"
output_dir = "./data/laion_dataset_nocap"




if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Create dataset from parquet files
dataset = datasets.load_dataset("parquet", data_files="data/laion/laion-high-resolution/*.parquet", cache_dir="data/laion/cache", split="train")

# Indices for each file
indices = [(i,min(i+num_files_per_par, len(dataset)-1)) for i in range(0, len(dataset), num_files_per_par)]
indices = {i:j for i,j in enumerate(indices)}



def download_image(url, r=0):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # If there is an HTTPError, return None
        if response.status_code == 404 or response.status_code == 403 or response.status_code == 402 or response.status_code == 401 or response.status_code == 400 or r > 5:
            return None
        else:
            # Try again
            return download_image(url, r+1)
    except requests.ConnectionError or requests.Timeout or requests.ConnectTimeout as e:
        if r > 5:
            return None
        else:
            # Try again
            return download_image(url, r+1)
    return response.content


def download_files(indices, index, dataset, output_dir, checkpoint_file_name, failed_file_name):
    # Read in the checkpoint file and see if this index is in it
    try:
        with open(checkpoint_file_name, "r") as f:
            checkpoint_indices = f.read().split("\n")[:-1]
            if str(index) in checkpoint_indices:
                return
    except:
        pass


    error_indices = []
    images = []

    # Iterate over all URLs
    for i in tqdm(range(indices[0], indices[1])):
        line = dataset[i]

        # Download file
        image = download_image(line["URL"])
        if image is None:
            error_indices.append(f"index: {index}, row: {i}, hash: {line['hash']}")
            continue

        images.append({
            "id": line["hash"], 
            "image": image,
            "pwatermark": line["pwatermark"],
            "punsafe": line["punsafe"],
        })

    # Save a parquet file
    df = pd.DataFrame(images)
    df.to_parquet(f"{output_dir}/data_{index}.parquet")

    # Wait until the checkpoint file is available
    while True:
        try:
            checkpoint_file = open(checkpoint_file_name, "a")
            break
        except Exception as e:
            pass
    # Append the index to the checkpoint file
    checkpoint_file.write(f"{index}\n")
    checkpoint_file.close()

    # Wait until the failed file is available
    while True:
        try:
            failed_file = open(failed_file_name, "a")
            break
        except:
            pass
    # Append the indices to the failed file
    for i in error_indices:
        failed_file.write(f"{i}\n")
    failed_file.close()

        

# Parallelize the download over all indices
# download_files(indices[0], 0, dataset, output_dir, checkpoint_file, failed_file)
with ThreadPoolExecutor() as executor:
    # Submit all tasks
    futures = [
        executor.submit(download_files, indices[idx], idx, dataset, output_dir, checkpoint_file, failed_file)
        for idx in indices.keys()
    ]

    # Wait for all tasks to complete
    for future in futures:
        # This will raise exceptions if any occur within the threads
        future.result()