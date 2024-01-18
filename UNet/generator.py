import os
import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from skimage.io import imread
from skimage.transform import resize


def get_batch(dataset_path: str, file_list: List[str], size: Tuple[int, int], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    n_batches = np.ceil(len(file_list) / batch_size)
    counter = 0
    data = []
    
    for file in tqdm(file_list, desc="Loading Images", unit="file"):
        image = imread(os.path.join(dataset_path, f'Combined/ImagesSmall/{file}.tif'))
        mask = imread(os.path.join(dataset_path, f"Combined/MasksSmall/{file}.tif"))
        
        image = (image / 255).astype(np.float32)
        mask = (mask / 255).astype(np.float32)
        mask = np.where(mask < 0.5, np.float32(0), np.float32(1))
        data.append([image, mask])

    while True:
        file_batch = data[batch_size * counter:batch_size * (counter + 1)]
        yield (np.array([f[0] for f in file_batch]), np.array([f[1] for f in file_batch]))
        counter += 1
        if counter == n_batches:
            random.shuffle(data)
            counter = 0
