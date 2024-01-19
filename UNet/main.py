"""Main entry point for training a new model.
The model is using the data that is placed in the folders
train and validation in the root directory of this repository.
The models will be saved to the models folder of the root directory.
"""
import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from metrics import jacc_coef, precision, recall, overall_accuracy
from model import create_model
from generator import get_batch


INPUT_SHAPE = (192, 192)
INPUT_CHANNELS = 4
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-8
DECAY_FACTOR = 0.7
PATIENCE = 10
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 8
MAX_NUM_EPOCHS = 50
L2_VALUE = 0.01
MEM_LIMIT_GPU = int(7.5 * 1024)

MODEL_PATH = '../model'

TRAINING_DATASET_PATH = '../train'
VALIDATION_DATASET_PATH = '../validation'

TRAINING_PATCHES = '../train/image_names_training.csv'
VALIDATION_PATCHES = '../validation/image_names_validation.csv'

files: List[str] = []
train_files: List[str] = []
test_files: List[str] = []
model_name: str = ""
model_idx: int = 1


def train():
    """Creates the model and starts the training process
    """

    # Create the generators for train and validation batches
    training_generator = get_batch(TRAINING_DATASET_PATH, train_files, INPUT_SHAPE, BATCH_SIZE)
    testing_generator = get_batch(VALIDATION_DATASET_PATH, test_files, INPUT_SHAPE, BATCH_SIZE)

    # Create the model and compile
    model = create_model(INPUT_SHAPE, INPUT_CHANNELS, 64, L2_VALUE)
    model.compile(
        optimizer=adam_v2.Adam(learning_rate=LEARNING_RATE, decay=WEIGHT_DECAY),
        loss='binary_crossentropy',
        metrics=['accuracy', jacc_coef, precision, recall, overall_accuracy]
    )
    print(model.summary())

    # Create all the callback functions
    model_checkpoint = ModelCheckpoint(
        f'../model_{model_idx}/weights_model_{model_idx}.hdf5',
        monitor='val_loss',
        mode='min', 
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )

    lr_reducer = ReduceLROnPlateau(
        factor=DECAY_FACTOR,
        cooldown=0,
        patience=PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1
    )

    csv_logger = CSVLogger(f'../model_{model_idx}/logs_model_{model_idx}.log')

    # Start the training of the model
    # It will automatically save the best model
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
    """Reads the training and validation file names into the variables
    """
    global train_files, test_files, files

    if not os.path.exists(TRAINING_PATCHES):
        print("Could not find the path to the csv with the image names")
        exit(1)

    with open(TRAINING_PATCHES, 'r', encoding='utf-8') as f:
        train_files = [line.strip() for line in f.readlines()[1:]]

    with open(VALIDATION_PATCHES, 'r', encoding='utf-8') as f:
        test_files = [line.strip() for line in f.readlines()[1:]]


def get_model_name():
    """Creates a new directory for the model that will
    be trained and sets the global model_idx variable to
    be the number of the trained model
    """
    global model_idx

    while True:
        if os.path.exists(os.path.join(MODEL_PATH, f'model_{model_idx}')):
            model_idx += 1
        break

    os.mkdir(os.path.join(MODEL_PATH, f'model_{model_idx}'))


def main():
    """Main function starts the training and sets the ENV variables for
    tensorflow.
    """
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Set memory growth for each GPU
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEM_LIMIT_GPU)])

    # Get the files from the disk
    get_files()

    # Get the name of the new model and create the folder
    get_model_name()

    # Train the model
    train()


if __name__ == '__main__':
    main()
