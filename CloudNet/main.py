import os
import random
import numpy as np
import tensorflow as tf
from typing import List
from datetime import datetime
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from metrics import jacc_coef, precision, recall, overall_accuracy
from model import create_model
from generator import get_batch
from utils import ADAMLearningRateTracker


INPUT_SHAPE = (192, 192)
INPUT_CHANNELS = 4
LEARNING_RATE = 1e-5
MIN_LEARNING_RATE = 1e-8
DECAY_FACTOR = 0.7
PATIENCE = 10
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 8
MAX_NUM_EPOCHS = 50
TEST_RATIO = 0.2
L2_VALUE = 0.01
RANDOM_STATE = 42
SHUFFLE_DATA = True
MEM_LIMIT_GPU = int(7.5 * 1024)

timestamp = datetime.strftime(datetime.now(), format='%y-%m-%d-%H-%M')
DATASET_PATH = '/home/kevin/Downloads/CloudNetDataset'
TRAINING_PATCHES = '/home/kevin/Downloads/95Cloud/95-cloud_training/training_patches_95-cloud_nonempty.csv' # 'training_patches_cleaned.csv'
WEIGHTS_PATH = f'weights-{timestamp}.hdf5'

files: List[str] = []
train_files: List[str] = []
test_files: List[str] = []


def train():

    model = create_model(INPUT_SHAPE, INPUT_CHANNELS, 64, L2_VALUE)
    model.compile(optimizer=adam_v2.Adam(learning_rate=LEARNING_RATE, decay=WEIGHT_DECAY), loss='binary_crossentropy', metrics=['accuracy', jacc_coef, precision, recall, overall_accuracy])
    print(model.summary())

    training_generator = get_batch(DATASET_PATH, train_files, INPUT_SHAPE, BATCH_SIZE)
    testing_generator = get_batch(DATASET_PATH, test_files, INPUT_SHAPE, BATCH_SIZE)


    model_checkpoint = ModelCheckpoint(f'../.checkpoints/{WEIGHTS_PATH}',
                                        monitor='val_loss',
                                        mode='min', 
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True)
    
    lr_reducer = ReduceLROnPlateau(factor=DECAY_FACTOR, cooldown=0, patience=PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1)
    csv_logger = CSVLogger(f'./model-{timestamp}.log')

    model.fit(
        training_generator,
        steps_per_epoch=np.ceil(len(train_files) / BATCH_SIZE),
        epochs=MAX_NUM_EPOCHS,
        verbose=1,
        callbacks=[model_checkpoint, lr_reducer, csv_logger],
        validation_data=testing_generator,
        validation_steps=BATCH_SIZE
    )


def get_files():
    global train_files, test_files, files

    if not os.path.exists(TRAINING_PATCHES):
        print("Could not find the path to the csv with the image names")
        exit(1)


    with open(TRAINING_PATCHES, 'r') as f:
        files = [line.strip() for line in f.readlines()[1:]]

    if SHUFFLE_DATA:
        random.shuffle(files)

    # For the first try, don't load all data
    # files = files[:500]

    # Splitting in training and test data
    split_idx = int(len(files) * (1 - TEST_RATIO))

    train_files = files[:split_idx]
    test_files = files[split_idx:]
    

def main():
    
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Set memory growth for each GPU
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEM_LIMIT_GPU)])
    
    # Get the files from the disk
    get_files()
    
    # Train the model
    train()
    

if __name__ == '__main__':
    main()
