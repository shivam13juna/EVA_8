import os
from PIL import Image

# define the input and output directories
input_dir = './'
output_file = 'file.txt'

# open the output file for writing
with open(output_file, 'w') as f:
    # loop over all the files in the input directory
    for filename in os.listdir(input_dir):
        # make sure the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue

        # open the image and get its dimensions
        filepath = os.path.join(input_dir, filename)
        with Image.open(filepath) as img:
            width, height = img.size

        # write the dimensions to the output file
        f.write(f'{filename} {width} {height}\n')
