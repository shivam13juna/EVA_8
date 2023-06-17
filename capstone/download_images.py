import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

df = pd.read_parquet(
    "part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
)
df.columns = [i.lower() for i in df.columns]

# Set the download directory where the images will be saved
download_dir = "image_dataset"
os.chdir(download_dir)

total_samples = min(100000, len(df))

# Randomly sample 60,000 rows from the dataframe
sample_df = df.sample(n=total_samples, random_state=42)

# Get the total number of samples


# Create a tqdm progress bar
progress_bar = tqdm(total=total_samples, unit="image")

def download_image(index):
    row = sample_df.loc[index]
    image_url = row["url"]
    prompt = row["text"]

    # Generate the image name based on the prompt
    image_name = prompt.replace(" ", "__") + ".jpg"

    try:
        # Download the image with a timeout of 5 seconds
        response = requests.get(image_url, timeout=5)
        
        # Check if the response status code indicates a successful download
        if response.status_code == 200:
            # Save the image
            with open(os.path.join(image_name), "wb") as file:
                file.write(response.content)
            progress_bar.update(1)
        else:
            pass
            # print(f"Error downloading image: {image_url}")
    except Exception as e:
        pass
        # print(f"Exception occurred while downloading image: {image_url}")
        # print(f"Exception details: {str(e)}")

# Using ThreadPoolExecutor to download images concurrently
with ThreadPoolExecutor(max_workers=40) as executor:
    executor.map(download_image, sample_df.index[:total_samples])

# Close the progress bar
progress_bar.close()

print("Image download complete.")
