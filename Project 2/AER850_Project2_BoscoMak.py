""" AER850 Project 2 - Bosco Mak"""


import numpy as np
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator



""" 1. Data Processing """

# set dimensions
w = 500
h = 500

batch_size = 32

# directories
train_dir = 'Data/train'
test_dir = 'Data/test'
val_dir = 'Data/valid'

# training data
train = keras.utils.image_dataset_from_directory(
    directory = train_dir,
    labels = "inferred",
    label_mode = 'categorical',
    image_size = (w, h),
    seed = 21,
    )
# test Data
test = keras.utils.image_dataset_from_directory(
    directory = test_dir,
    labels = "inferred", 
    label_mode = 'categorical',
    image_size = (w, h),
    seed = 21,
    )

# validation data
val = keras.utils.image_dataset_from_directory(
    directory = val_dir,
    labels = "inferred",
    label_mode = 'categorical',
    image_size = (w, h),
    seed = 21,
    )

# Data augmentation 

rescale = layers.Rescaling(1./255)             # scale to range [0, 255]

data_augmentation = keras.Sequential([
    layers.RandomZoom(0.2),                    # zoom range
    layers.RandomShear(0.2),                   # shear range 
])

# Apply normalization and augmentation to the training dataset
train = train.map(lambda x, y: (rescale(data_augmentation(x, training=True)), y)) # lambda - 

# Apply only rescaling to validation and test datasets
val = val.map(lambda x, y: (rescale(x), y))
test = test.map(lambda x, y: (rescale(x), y))


""" 2. Neural Network Architecture Design """



mdl1 = keras.Sequential([
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu", input_shape=(w, h, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),                 
    layers.Flatten(),                           # Flatten input
    layers.Dense(64, activation="relu"),        # Relu hidden layers, 64 neurons
    layers.Dropout(0.3),                        # 30% dropout to prevent overfitting
    layers.Dense(3, activation="softmax")       # output layer, 3 classes
])

mdl1.summary()


""" 3. Hyperparameter Analysis """

""" 4. Model Evaluation """

""" 5. Model Testing """

