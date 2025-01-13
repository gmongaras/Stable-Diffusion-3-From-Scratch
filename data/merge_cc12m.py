import json
import os
import jsonlines
import pandas as pd
import pyarrow.parquet as pq
import dask.dataframe as dd


delete_while_merging = True
error_file = "data/errors.txt"
out_dir = "data/final_dataset"


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# # Load in train.jsonl
# cc12m_captions_dataset = []
# with jsonlines.open("data/cc12m_captions/train.jsonl", "r") as reader:
#     for i, l in enumerate(reader):
#         cc12m_captions_dataset.append(l)
#         if i > 1000:
#             break
cc12m_captions_dataset = dd.read_json("data/cc12m_captions/train.jsonl", lines=True, dtype={"key": str})[["caption_llava_short", "key"]].compute()


# Convert cc12m_captions_dataset to a dictionary for faster lookups
caption_dict = cc12m_captions_dataset.set_index("key")["caption_llava_short"].to_dict()

# Open the error file
error_file = open(error_file, "a")


# Used to process a single row
def process_row(row):
    try:
        # Look up the caption
        caption = caption_dict.get(row["id"], None)
        if caption is None:
            caption = row["caption"][1]["value"]
            error_file.write(f"file: {filename}, line: {row.name}, id: {row['id']}\n")
        row["caption"] = caption.strip()
    except Exception as e:
        error_file.write(f"Error: {e}, file: {filename}, line: {row.name}, id: {row['id']}\n")
    
    # Update image
    row["image"] = row["image"]["bytes"]
    return row


# Iterate through all files in the data/cc12m/data directory
while len(os.listdir("data/cc12m/data")) > 0:
    print("Processing file. Remaining files:", len(os.listdir("data/cc12m/data")))

    # Load in the next file
    filename = os.listdir("data/cc12m/data")[0]
    file = pd.read_parquet("data/cc12m/data/" + filename)[["id", "image", "conversations"]]
    file.rename(columns={"conversations": "caption"}, inplace=True)

    # Change the captions for each image
    file = file.apply(process_row, axis=1)
    
    # Save the file to the out_dir
    file.to_parquet(out_dir + "/" + filename)
    # Delete the original file
    if delete_while_merging:
        os.remove("data/cc12m/data/" + filename)
    

error_file.close()