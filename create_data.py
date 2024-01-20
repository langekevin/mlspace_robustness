"""Script combines the single items from the 38-Cloud and 95-Cloud datasets
This results in a single dataset containing all available images and masks
Assuming the images are separated in two locations - the folder structure
would be as follows

38Cloud
  - training
    - train_red
    - train_green
    - train_blue
    - train_nir
    - train_gt
95Cloud
  - training
    - train_red
    - train_green
    - train_blue
    - train_nir
    - train_gt
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from skimage import io


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-38cloud', type=str, required=True, help="Path to the training folder of the 38-Cloud dataset")
parser.add_argument('--dataset-95cloud', type=str, required=True, help="Path to the training folder of the 95-Cloud dataset")
parser.add_argument('--train-images', type=str, required=True, help="Image names of the training data")
parser.add_argument('--validation-images', type=str, required=True, help="Image names of the validation data")
args = parser.parse_args()


def get_images(path: str, target: str):
    """Function loads the images from the source path to the final structure
    of how the data will be expected later in the training and testing process

    Args:
        path (str): The path to the files containing the image names
        target (str): The path where the images need to be saved
    """
    if not os.path.exists(path):
        print("Path not available")
        return

    with open(path, 'r', encoding='utf-8') as f:
        image_names = [x.strip() for x in f.readlines()[1:]]

    for image_name in tqdm(image_names, desc="Loading images", unit="images"):
        if os.path.exists(os.path.join(args.dataset_38cloud, f'train_red/red_{image_name}.TIF')):
            file_path = args.dataset_38cloud
        elif os.path.exists(os.path.join(args.dataset_95cloud, f'train_red/red_{image_name}.TIF')):
            file_path = args.dataset_95cloud
        else:
            print("Path invalid")
            return

        img_red = io.imread(os.path.join(file_path, f'train_red/red_{image_name}.TIF'))
        img_green = io.imread(os.path.join(file_path, f'train_green/green_{image_name}.TIF'))
        img_blue = io.imread(os.path.join(file_path, f'train_blue/blue_{image_name}.TIF'))
        img_nir = io.imread(os.path.join(file_path, f'train_nir/nir_{image_name}.TIF'))

        img = np.stack((img_red, img_green, img_blue, img_nir), axis=-1) / 65535 * 255
        img = img.astype(np.uint8)
        io.imsave(os.path.join(target, f'input/{image_name}.TIF'), img, check_contrast=False)

        img_gt = io.imread(os.path.join(file_path, f'train_gt/gt_{image_name}.TIF')).astype(np.uint8)
        io.imsave(os.path.join(target, f'gt/{image_name}.TIF'), img_gt, check_contrast=False)


def check_if_folder_exists() -> bool:
    """Checks if the given path is existend

    Returns:
        bool: True if data was found, otherwise False
    """
    if not os.path.exists(args.dataset_38cloud):
        return False
    if not os.path.exists(args.dataset_95cloud):
        return False

    for folder in ['train_red', 'train_blue', 'train_green', 'train_nir', 'train_gt']:
        if folder not in os.listdir(args.dataset_38cloud):
            return False
        if folder not in os.listdir(args.dataset_95cloud):
            directory_content = os.listdir(args.dataset_95cloud)
            for item in directory_content:
                if str(item).startswith(folder):
                    os.rename(
                        src=os.path.join(args.dataset_95cloud, item),
                        dst=os.path.join(args.dataset_95cloud, folder)
                    )
                    break
            else:
                return False
    
    # Check if the target folders exist (e. g. input and gt)
    if not os.path.exists('train/input'):
        os.mkdir('train/input')
    if not os.path.exists('train/gt'):
        os.mkdir('train/gt')
    
    if not os.path.exists('validation/input'):
        os.mkdir('validation/input')
    if not os.path.exists('validation/gt'):
        os.mkdir('validation/gt')

    return True


def main():
    """Main method initiates loading of images for training and validation"""

    folder_exists = check_if_folder_exists()
    if not folder_exists:
        print("[E] The data of the 38Cloud and/or 95Cloud dataset was not found!")
        exit(1)

    get_images(args.train_images, 'train')
    get_images(args.validation_images, 'validation')


if __name__ == '__main__':
    main()
