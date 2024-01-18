# Script combines the single items from the 38-Cloud and 95-Cloud datasets
# This results in a single dataset containing all available images and masks
# Assuming the images are separated in two locations - the folder structure
# would be as follows
#
# 38Cloud
#   - training
#     - train_red
#     - train_green
#     - train_blue
#     - train_nir
#     - train_gt
# 95Cloud
#   - training
#     - train_red
#     - train_green
#     - train_blue
#     - train_nir
#     - train_gt

import os
import numpy as np
from tqdm import tqdm
from skimage import io


path_38cloud_root = 'PATH/TO/95CLOUD/DATASET'
path_95cloud_root = 'PATH/TO/38CLOUD/DATASET'

path_training_image_names = './train/image_names_training.csv'
path_validation_image_names = './validation/image_names_validation.csv'


def get_images(path: str, target: str):
    if not os.path.exists(path):
        print("Path not available")
        return

    with open(path, 'r') as f:
        image_names = [x.strip() for x in f.readlines()[1:]]
    
    for image_name in tqdm(image_names, desc="Loading images", unit="images"):
        if os.path.exists(os.path.join(path_38cloud_root, f'training/train_red/red_{image_name}.tif')):
            file_path = path_38cloud_root
        elif os.path.exists(os.path.join(path_95cloud_root, f'training/train_red/red_{image_name}.tif')):
            file_path = path_95cloud_root
        else:
            print("Path invalid")
            return
        
        img_red = io.imread(os.path.join(file_path, f'training/train_red/red_{image_name}.tif'))
        img_green = io.imread(os.path.join(file_path, f'training/train_green/green_{image_name}.tif'))
        img_blue = io.imread(os.path.join(file_path, f'training/train_blue/blue_{image_name}.tif'))
        img_nir = io.imread(os.path.join(file_path, f'training/train_nir/nir_{image_name}.tif'))

        img = np.stack((img_red, img_green, img_blue, img_nir), axis=-1) / 65535 * 255
        img = img.astype(np.uint8)
        io.imsave(os.path.join(target, f'input/{image_name}.tif'), img, check_contrast=False)
        
        img_gt = io.imread(os.path.join(file_path, f'training/train_gt/gt_{image_name}.tif')).astype(np.uint8)
        io.imsave(os.path.join(target, f'gt/{image_name}.tif'), img_gt, check_contrast=False)

def main():
    get_images(path_training_image_names, 'train')
    get_images(path_validation_image_names, 'validation')

if __name__ == '__main__':
    main()
