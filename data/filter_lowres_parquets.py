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
PngImagePlugin.MAX_TEXT_CHUNK = 999999 * (1024**2)




input_dir = "data/cc12m_and_imagenet21K/"
output_dir = "data/cc12m_and_imagenet21K_highqual/"
img_col = "image"




REPEATED_OPENINGS = [
    ('the image showcases ', ''),
    ('the image portrays ', ''),
    ('the image appears to be ', ''),
    ('the image is ', ''),
    ('the image depicts ', ''),
    ('the image features ', ''),
    ('the image captures ', ''),
    ('the image shows ', ''),
    ('the image displays ', ''),
    ('the image presents ', ''),
    ('this image showcases ', ''),
    ('this image portrays ', ''),
    ('this image appears to be ', ''),
    ('this image is ', ''),
    ('this image depicts ', ''),
    ('this image features ', ''),
    ('this image captures ', ''),
    ('this image shows ', ''),
    ('this image displays ', ''),
    ('this image presents ', ''),
    ('in this picture, ', ''),
    ('in this artwork, ', 'artwork of '),
    ('in this illustration, ', 'illustration of '),
    ('in this depiction, ', ''),
    ('in this piece, ', ''),
    ('in this image, ', ''),
    ('in this art piece, ', 'art of '),
    ('in this scene, ', ''),
    ('in the picture, ', ''),
    ('in the artwork, ', 'artwork of '),
    ('in the illustration, ', 'illustration of '),
    ('in the depiction, ', ''),
    ('in the piece, ', ''),
    ('in the image, ', ''),
    ('in the art piece, ', 'art of '),
    ('in the scene, ', ''),
]




if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Total number of rows and total dropped
total_rows = 0
total_dropped = 0

# Iterate over all the files in the input directory
for i, file in enumerate(tqdm(os.listdir(input_dir))):
    if not file.endswith(".parquet"):
        continue

    # Skip is parquet file is already processed, that is in the output directory
    if file in os.listdir(output_dir):
        continue

    print(f"Processing {file}, index number {i}...")

    # Load the parquet file
    df = pd.read_parquet(os.path.join(input_dir, file))
    initial_size = len(df)

    # Add empty columns "height", "width", and "aspect_ratio"
    df["height"] = None
    df["width"] = None
    df["aspect_ratio"] = None

    # For each row, load the image from bytes and get the height, width, and aspect ratio
    for index, row in df.iterrows():
        try:
            # Load the image from bytes
            image = Image.open(io.BytesIO(row[img_col]))

            # Get the height and width of the image
            height, width = image.size

            # Calculate the aspect ratio
            aspect_ratio = width / height

            # Update the dataframe
            df.at[index, "height"] = height
            df.at[index, "width"] = width
            df.at[index, "aspect_ratio"] = aspect_ratio

        except Exception as e:
            print(f"Error processing image at index {index}: {e}")

    # Drop any rows where the image could not be loaded
    df = df.dropna(subset=["height", "width", "aspect_ratio"])

    # Drop any rows where the image is not high quality. That is, if both the height and width are less than 256.
    df = df[(df["height"] >= 256) | (df["width"] >= 256)]

    # Replace the repeated openings with an empty string
    for opening, replacement in REPEATED_OPENINGS:
        df["recaption"] = df["recaption"].str.replace(opening, replacement, regex=False)
        df["recaption_short"] = df["recaption_short"].str.replace(opening, replacement, regex=False)

    # Capitalize the first letter of the caption
    df["recaption"] = df["recaption"].str.capitalize()
    df["recaption_short"] = df["recaption_short"].str.capitalize()

    # Drop any rows where the text ("recaption" or "recaption_short") are empty or less than 10 characters
    df = df[(df["recaption"].str.len() >= 10) & (df["recaption_short"].str.len() >= 10)]

    # Reset the index
    df = df.reset_index(drop=True)

    # Save the dataframe to a new parquet file
    output_file = os.path.join(output_dir, file)
    df.to_parquet(output_file, index=False)

    # Print the number of rows dropped
    print(f"Dropped {initial_size - len(df)} rows or {100 - (len(df) / initial_size * 100):.2f}% of the original dataset.")

    # Increment the total number of rows and total dropped
    total_rows += len(df)
    total_dropped += initial_size - len(df)


print(f"Total rows: {total_rows}")
print(f"Total dropped: {total_dropped}")
print(f"Total dropped percentage: {total_dropped / (total_rows + total_dropped) * 100:.2f}%")