import os
import tarfile
import pandas as pd
from pathlib import Path
import json
from PIL import Image
import io
from concurrent.futures import ProcessPoolExecutor

def image_to_bytes(image):
    rbuffer = io.BytesIO()
    image.save(rbuffer, format="PNG")
    return rbuffer.getvalue()

def process_tar_file(tar_file, extract_path, output_path, class_mapping):
    # Create the extraction folder
    extract_path = Path(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Processing {tar_file.name}...")

        # Extract .tar contents
        with tarfile.open(tar_file) as tar:
            tar.extractall(path=extract_path)

        # Output dataframe with the image in bytes and class
        output_frame = []

        # Iterate over all images in the extracted file
        for file in extract_path.glob("*"):
            try:
                image = Image.open(file).convert("RGB")
                # Convert the image to raw bytes
                image = image_to_bytes(image)
                # Append the image and class to the output dataframe
                output_frame.append({
                    "image": image,
                    "class": class_mapping[file.stem.split("_")[0]],
                    "id": file.stem
                })
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
            
        # Convert the output frame to a DataFrame
        df = pd.DataFrame(output_frame)

        # Save the DataFrame as a parquet file
        parquet_file = output_path / f"{tar_file.stem}.parquet"
        df.to_parquet(parquet_file, index=False)

        print(f"Converted {tar_file.name} to {parquet_file.name}")

        # Delete all files in the extracted folder
        for file in extract_path.glob("*"):
            os.remove(file)

        # Delete the tar file
        os.remove(tar_file)

    except Exception as e:
        print(f"Error processing {tar_file.name}: {e}")

    # Delete the extraction folder
    extract_path.rmdir()

def extract_tar_to_parquet(input_dir, output_dir, extract_subfolder_name="extracted"):
    """
    Converts all .tar files in a folder to .parquet files by extracting their contents.

    Parameters:
        input_dir (str): Path to the folder containing .tar files.
        output_dir (str): Path to the folder where .parquet files will be saved.
        extract_subfolder_name (str): Subfolder name to temporarily store extracted files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    extract_path = input_path / extract_subfolder_name

    # Create output and temporary extraction folders
    output_path.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Load in the class mapping
    with open("data/imagenet21_class_to_string.json", "r") as f:
        class_mapping = json.load(f)

    # Ensure paths exist
    input_path = Path(input_path)
    extract_path = Path(extract_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tar_files = list(input_path.glob("*.tar"))
    if not tar_files:
        print("No .tar files found in the input directory.")
        return

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_tar_file, tar_file, f"{extract_path}_{i}", output_path, class_mapping
            )
            for i, tar_file in enumerate(tar_files)
        ]
        
        for future in futures:
            try:
                future.result()  # Wait for each task to complete
            except Exception as e:
                print(f"Error in parallel processing: {e}")

    # Remove temporary extraction folder
    try:
        extract_path.rmdir()
    except Exception as e:
        print(f"Error removing extraction folder: {e}")

    print("Conversion complete.")

# Example usage:
input_folder = "data/Imagenet21"  # Replace with the folder containing .tar files
output_folder = "data/Imagenet21"  # Replace with the desired output folder
extract_tar_to_parquet(input_folder, output_folder)