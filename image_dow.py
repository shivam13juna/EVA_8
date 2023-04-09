import os
import requests
from PIL import Image
from io import BytesIO
from google_images_search import GoogleImagesSearch

# Replace these with your own API key and Custom Search Engine ID
api_key = "AIzaSyD-71IPdzLjA7AEtP3YHMgoyfNu2I2bXSY"
cse_id = "571c9d1f874e74df6"

# Set up the Google Images Search object
gis = GoogleImagesSearch(api_key, cse_id)

# Set up the YOLO directory structure
root_dir = 'yolo_images/Elsa Frozen 1'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

topic = 'Elsa Frozen 1'  # Replace this with your desired topic

# Configure the search parameters
search_params = {
    'q': topic,
    'num': 100,  # Number of images to fetch. Max is 10 for Free API.
    'imgSize': 'large',
    'fileType': 'jpg|png',
}

# Perform the search
gis.search(search_params)

# Download and resize the images
for i, image in enumerate(gis.results()):
    try:
        response = requests.get(image.url)
        img = Image.open(BytesIO(response.content))

        # Resize the image while maintaining the aspect ratio
        
        # Save the image in the YOLO directory structure
        img_format = 'JPEG' if img.mode == 'RGB' else 'PNG'
        img_filename = f"{root_dir}/{topic}_{i:03d}.{img_format.lower()}"
        img.save(img_filename, format=img_format)

        print(f"Image saved as {img_filename}")
    except:
        print(f"Failed to download image {i}")
