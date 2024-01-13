from typing import Tuple
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate
from tensorflow.python.keras.regularizers import l2


def create_model(size: Tuple[int, int], channels: int, start_neurons: int, l2_value: float = 0.005):
    inputs = Input((size[0], size[1], channels))
    conv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(inputs)
    conv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(pool1)
    conv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(pool2)
    conv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(pool3)
    conv4 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(start_neurons * 16, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(pool4)
    conv5 = Conv2D(start_neurons * 16, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv5)

    up6 = Conv2DTranspose(start_neurons * 8, 2, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv5)
    merge6 = concatenate([conv4, up6], axis = 3)
    merge6 = Dropout(0.5)(merge6)
    conv6 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(merge6)
    conv6 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv6)

    up7 = Conv2DTranspose(start_neurons * 4, 2, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    merge7 = Dropout(0.5)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv7)

    up8 = Conv2DTranspose(start_neurons * 2, 2, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    merge8 = Dropout(0.5)(merge8)
    conv8 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(merge8)
    conv8 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv8)

    up9 = Conv2DTranspose(start_neurons * 4, 2, strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(l2_value))(conv8)
    merge9 = concatenate([conv1,up9], axis=3)
    merge9 = Dropout(0.5)(merge9)
    conv9 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_value))(merge9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    return Model(inputs=inputs, outputs=outputs)
