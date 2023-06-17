import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('image_data.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        try:
            # Read the images using cv2
            source_cv = cv2.imread(source_filename)
            target_cv = cv2.imread(target_filename)

            # Convert color space from BGR to RGB
            source_cv = cv2.cvtColor(source_cv, cv2.COLOR_BGR2RGB)
            target_cv = cv2.cvtColor(target_cv, cv2.COLOR_BGR2RGB)

            # Resize images
            source_cv = cv2.resize(source_cv, (512, 512))
            target_cv = cv2.resize(target_cv, (512, 512))

        except:
            print('Error reading image.')
            return None

        # Normalize source images to [0, 1].
        source = source_cv.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target_cv.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
