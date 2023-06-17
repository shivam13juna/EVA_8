import os
import cv2
import warnings
from tqdm import tqdm

directory = "dataset/target/"


sorted_filesnames = sorted(os.listdir(directory), key = lambda x: int(x.split('.')[0]))

for filename in sorted_filesnames:
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            print(filename)
            img = cv2.imread(os.path.join(directory, filename))
            
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename) # print out the names of corrupt files

