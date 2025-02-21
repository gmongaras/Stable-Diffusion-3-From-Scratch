# Change huggingface directory
import os
os.environ["HF_HOME"] = "data/cache"
import multiprocessing
from multiprocessing import Process
from PIL import Image
import pandas as pd
import io
from tqdm import tqdm
from PIL import PngImagePlugin
import concurrent.futures
PngImagePlugin.MAX_TEXT_CHUNK = 999999 * (1024**2)




input_dir = "data/cc12m_and_imagenet21K_highqual/"
max_resolution = 256
patch_size = 16
output_dir = f"data/cc12m_and_imagenet21K_highqual_{max_resolution}/"





def image_to_bytes(image):
    rbuffer = io.BytesIO()
    image.save(rbuffer, format="PNG")
    return rbuffer.getvalue()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# # Iterate over all the files in the input directory
# for i, file in enumerate(tqdm(os.listdir(input_dir))):
#     if not file.endswith(".parquet"):
#         continue

#     # Skip is parquet file is already processed, that is in the output directory
#     if file in os.listdir(output_dir):
#         continue

#     print(f"Processing {file}, index number {i}...")

#     # Load the parquet file
#     df = pd.read_parquet(os.path.join(input_dir, file))

#     # Add empty columns "bucket_size"
#     df["bucket_size"] = None

#     # For each row, load the image from bytes and get the height, width, and aspect ratio
#     for index, row in df.iterrows():
#         try:
#             # Load the image from bytes
#             image = Image.open(io.BytesIO(row["image"]))

#             # Get the height and width of the image
#             height, width = image.size

#             # If the width or height is greater than 256, resize the image to have the largest side be 256
#             if width > max_resolution or height > max_resolution:
#                 # If width greater than height, resize width to 256 and scale height accordingly, then to a factor of the patch size
#                 if width > height:
#                     new_width = max_resolution
#                     new_height = int(height * (max_resolution / width))
#                     # Scale to whichever factor of the patch size is closer
#                     remainder = new_height % patch_size
#                     if patch_size - remainder < remainder:
#                         remainder = patch_size - remainder
#                     new_height = new_height + remainder
#                 # If height greater than width, resize height to 256 and scale width accordingly, then to a factor of the patch size
#                 else:
#                     new_height = max_resolution
#                     new_width = int(width * (max_resolution / height))
#                     # Scale to whichever factor of the patch size is closer
#                     remainder = new_width % patch_size
#                     if patch_size - remainder < remainder:
#                         remainder = patch_size - remainder
#                     new_width = new_width + remainder


#                 # Resize the image
#                 image = image.resize((new_height, new_width), resample=Image.LANCZOS)

#                 # Update the height and width
#                 height, width = image.size
#                 df.at[index, "height"] = height
#                 df.at[index, "width"] = width
#                 df.at[index, "aspect_ratio"] = width / height

#                 # Bucket size is (height, width)
#                 df.at[index, "bucket_size"] = f"{height}x{width}"

#                 # Update the image
#                 df.at[index, "image"] = image_to_bytes(image)

#         except Exception as e:
#             print(f"Error processing image at index {index}: {e}")

#             # Make the height, width, and aspect ratio None
#             df.at[index, "height"] = df.at[index, "width"] = df.at[index, "aspect_ratio"] = df.at[index, "image"] = None

#     # Drop any rows where the image could not be loaded
#     df = df.dropna(subset=["height", "width", "aspect_ratio", "image"])

#     # Reset the index
#     df = df.reset_index(drop=True)

#     # Save the dataframe to a new parquet file
#     output_file = os.path.join(output_dir, file)
#     df.to_parquet(output_file, index=False)


def process_file(file):
    """Process a single Parquet file."""
    output_file = os.path.join(output_dir, file)
    
    # Skip if already processed
    if os.path.exists(output_file):
        return

    print(f"Processing {file}...")

    # Load the Parquet file
    df = pd.read_parquet(os.path.join(input_dir, file))

    # Add an empty "bucket_size" column
    df["bucket_size"] = None

    for index, row in df.iterrows():
        try:
            # Load the image from bytes
            image = Image.open(io.BytesIO(row["image"]))

            # Get image dimensions
            height, width = image.size

            # Resize logic
            # If one edge is larger than the max resolution, we need to downscale
            if width > max_resolution or height > max_resolution:
                if width > height:
                    # Resize width to max_resolution and scale height accordingly, then to a factor of the patch size
                    new_width = max_resolution
                    new_height = int(height * (max_resolution / width))
                    remainder = new_height % patch_size
                    new_height += patch_size - remainder if patch_size - remainder < remainder else -remainder
                else:
                    # Resize height to max_resolution and scale width accordingly, then to a factor of the patch size
                    new_height = max_resolution
                    new_width = int(width * (max_resolution / height))
                    remainder = new_width % patch_size
                    new_width += patch_size - remainder if patch_size - remainder < remainder else -remainder
            # If none are larger than the max resolution, just make sure they are factors of the patch size
            else:
                # Resize both the neight and width to the nearest factor of the patch size
                height_remainder = height % patch_size
                new_height = height + (patch_size - height_remainder if patch_size - height_remainder < height_remainder else -height_remainder)
                width_remainder = width % patch_size
                new_width = width + (patch_size - width_remainder if patch_size - width_remainder < width_remainder else -width_remainder)

            image = image.resize((new_height, new_width), resample=Image.Resampling.LANCZOS)
            height, width = image.size
            assert height % patch_size == 0 and width % patch_size == 0, f"Image dimensions {height}x{width} are not factors of the patch size {patch_size}."
            df.at[index, "height"] = height
            df.at[index, "width"] = width
            df.at[index, "aspect_ratio"] = width / height
            df.at[index, "bucket_size"] = f"{height}x{width}"
            df.at[index, "image"] = image_to_bytes(image)

        except Exception as e:
            print(f"Error processing image at index {index}: {e}")
            df.at[index, ["height", "width", "aspect_ratio", "image"]] = None

    # Drop rows where image loading failed
    df = df.dropna(subset=["height", "width", "aspect_ratio", "image"]).reset_index(drop=True)

    # The length should not be 0, but if it is, skip saving
    if len(df) == 0:
        print(f"Skipping {file} due to no valid images.")
        return

    # Save to output directory
    df.to_parquet(output_file, index=False)

    print(f"Finished processing {file} with {len(df)} rows.")

    del df

"""Parallelize file processing."""
files = [file for file in os.listdir(input_dir) if file.endswith(".parquet")]

with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
    futures = {executor.submit(process_file, file): file for file in files}
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(files)):
        try:
            future.result()  # Raise exception if occurred inside process
        except Exception as e:
            failed_file = futures[future]
            print(f"Process failed for file {failed_file}: {e}")