import os
import random
from typing import List, Tuple, Generator
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize


def get_batch(dataset_path: str, file_list: List[str], size: Tuple[int, int], batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """The function first loads all data from the disk into the memory to then create
    batches of a specified batch_size during the training and validation process.

    Args:
        dataset_path (str): The path where the dataset is located
        file_list (List[str]): The list of filenames that need to be loaded
        size (Tuple[int, int]): The target dimensions of the images (resolution)
        batch_size (int): The batch size that was used for the training process

    Yields:
        Generator[Iterator[Tuple[np.ndarray, np.ndarray]]]: Yields an iterator that creates
            tuples where each tuple consists of the input images on the first index
            and the ground truth on the second image
    """
    n_batches = np.ceil(len(file_list) / batch_size)
    counter = 0
    data = []

    for file in tqdm(file_list, desc="Loading Images", unit="file"):
        image = imread(os.path.join(dataset_path, f'Combined/ImagesSmall/{file}.tif'))
        mask = imread(os.path.join(dataset_path, f"Combined/MasksSmall/{file}.tif"))

        image = resize(image, size, preserve_range=True, mode='symmetric')
        mask = resize(mask, size, preserve_range=True, mode='symmetric')

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
