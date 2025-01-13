import pandas as pd
import os
import tarfile
import shutil
from PIL import Image
import io
import json
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq


input_folder = "data/laion/relaion2B-en-research-data"
output_folder = "data/laion/relaion2B-en-research-data-sharded"
tmp_base = "data/laion/tmp"
num_tar_per_shard = 25





def image_to_bytes(image):
    rbuffer = io.BytesIO()
    image.save(rbuffer, format="PNG")
    return rbuffer.getvalue()


# Get all parquet files in the input folder
parquet_files = [f"{input_folder}/{f}" for f in os.listdir(input_folder) if f.endswith(".parquet")]
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(tmp_base):
    os.makedirs(tmp_base)



def process_tar_file(tar_file):
    try:
        # Untar the corresponding tar files
        with tarfile.open(tar_file) as tar:
            # If the tar directory exists, delete it
            pth = os.path.join(tmp_base, os.path.basename(tar_file).replace(".tar", ""))
            if os.path.exists(pth):
                shutil.rmtree(pth)
            tar.extractall(pth)

        # Get all .json and .png in the extracted tar directory
        files = [f"{pth}/{f}".replace(".png", "") for f in os.listdir(pth) if f.endswith(".png")]

        def process_file(file):
            try:
                # Load the JSON file and PNG
                try:
                    with open(f"{file}.json") as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"Error loading JSON file {file}.json: {e}")
                    return None
                try:
                    img = Image.open(f"{file}.png")
                except Exception as e:
                    print(f"Error loading image file {file}.png: {e}")
                    return None

                # Skip if errors
                if metadata["status"] != "success":
                    return None

                # Convert image to raw bytes
                img_ = image_to_bytes(img)
                img.close()
                img = img_

                # Return the processed dictionary
                return {
                    "key": metadata["key"],
                    "image": img,
                    "caption": metadata["caption"],
                    "url": metadata["url"],
                    "punsafe": metadata["punsafe"],
                    "pwatermark": metadata["pwatermark"],
                    "similarity": metadata["similarity"]
                }
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_file, files), total=len(files)))

        # Filter out None results
        df_ = [result for result in results if result is not None]

        # Delete the extracted tar directory
        shutil.rmtree(pth)

        return df_

    except Exception as e:
        print(f"Error processing tar file {tar_file}: {e}")
        return []


# Function used to iterate over a given set of parquet files,
# untar each of the data files, combine into shards, and save them
def untar_and_shard(parquet_files, output_folder):
    df = []

    # Get all tar files
    tar_files = [parquet_file.replace(".parquet", ".tar") for parquet_file in parquet_files]

    # # Iterate over the parquet and tar files
    # for tar_file in tar_files:
    #     # Untar the corresponding tar files
    #     with tarfile.open(tar_file) as tar:
    #         # If the tar directory exists, delete it
    #         pth = os.path.join(tmp_base, os.path.basename(tar_file).replace(".tar", ""))
    #         if os.path.exists(pth):
    #             shutil.rmtree(pth)
    #         tar.extractall(pth)
    #     # Get all .json and .png in the extracted tar directory
    #     files = [f"{pth}/{f}".replace(".json", "") for f in os.listdir(pth) if f.endswith(".json")]


    #     # Function to process a single file
    #     def process_file(file):
    #         try:
    #             # Load the JSON file and PNG
    #             with open(f"{file}.json") as f:
    #                 metadata = json.load(f)
    #             img = Image.open(f"{file}.png")

    #             # Skip if errors
    #             if metadata["status"] != "success":
    #                 return None

    #             # Convert image to raw bytes
    #             img = image_to_bytes(img)

    #             # Return the processed dictionary
    #             return {
    #                 "key": metadata["key"],
    #                 "image": img,
    #                 "caption": metadata["caption"],
    #                 "url": metadata["url"],
    #                 "punsafe": metadata["punsafe"],
    #                 "pwatermark": metadata["pwatermark"],
    #                 "similarity": metadata["similarity"]
    #             }
    #         except Exception as e:
    #             print(f"Error processing file {file}: {e}")
    #             return None
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         # Wrap tqdm around executor.map for progress tracking
    #         results = list(tqdm(executor.map(process_file, files), total=len(files)))
    #     # Filter out None results
    #     df_ = [result for result in results if result is not None]
        
    #     # Add to the main dataframe
    #     df.extend(df_)

    #     # Delete the extracted tar directory
    #     shutil.rmtree(pth)

    # Parallelize the outer loop
    df = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_tar_file, tar_files), total=len(tar_files)))

    # Combine all results
    for result in results:
        df.extend(result)
        
    # Save the dataframe to a parquet file
    df = pd.DataFrame(df)
    df.to_parquet(f"{output_folder}/{os.path.basename(parquet_files[0])}", compression='snappy')

    # Delete all tar and parquet files
    for tar_file in tar_files:
        os.remove(tar_file)
    for parquet_file in parquet_files:
        os.remove(parquet_file)

    del df
        

# Group the parquet files into sets
parquet_files = [parquet_files[i:i+num_tar_per_shard] for i in range(0, len(parquet_files), num_tar_per_shard)]

# Untar and shard the data
for parquet_files_group in parquet_files:
    untar_and_shard(parquet_files_group, output_folder)
