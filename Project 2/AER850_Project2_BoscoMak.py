""" AER850 Project 2 - Bosco Mak"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator



""" 1. Data Processing """

# set dimensions/constants
w = 500
h = 500


tf.keras.utils.set_random_seed(21)
AUTOTUNE = tf.data.AUTOTUNE

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
    )

# test Data
test = keras.utils.image_dataset_from_directory(
    directory = test_dir,
    labels = "inferred", 
    label_mode = 'categorical',
    image_size = (w, h),
    )

# validation data
val = keras.utils.image_dataset_from_directory(
    directory = val_dir,
    labels = "inferred",
    label_mode = 'categorical',
    image_size = (w, h),
    )

# Sanity check
class_names = train.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Data augmentation 

rescale = layers.Rescaling(1./255)      # scale to range [0, 255]

data_augmentation = keras.Sequential([
    layers.RandomZoom(0.2),             # zoom range
    layers.RandomRotation(0.1),         # Rotation + Translation replaces shear
    layers.RandomTranslation(0.1, 0.1)
])

# Apply normalization and augmentation to the training dataset
train = train.map(lambda x, y: (rescale(data_augmentation(x, training=True)), y))

# Apply only rescaling to validation and test datasets
val = val.map(lambda x, y: (rescale(x), y))
test = test.map(lambda x, y: (rescale(x), y))


""" 2. Neural Network Architecture Design """

batch_size = 32
epoch = 15


early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
    )

mdl1 = keras.Sequential([
    layers.Conv2D(filters = 32, kernel_size = (3,3),activation="relu", input_shape=(w, h, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),                 
    layers.Flatten(),                           # Flatten input
    layers.Dense(128, activation="relu"),        # Relu hidden layers, 64 neurons
    layers.Dropout(0.3),                        # 30% dropout to prevent overfitting
    layers.Dense(3, activation="softmax")       # output layer, 3 classes
])

mdl1.summary()

mdl1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



history1 = mdl1.fit(
    train,
    validation_data=(val),
    epochs=epoch,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate CNN model
test_loss1, test_acc1 = mdl1.evaluate(test)
print(f"Test accuracy: {test_acc1:.4f} | Test loss: {test_loss1:.4f}")

# Plotting model

plt.figure(figsize=(6,4))
plt.plot(history1.history["accuracy"], label="Train Acc")
plt.plot(history1.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

""" 3. Hyperparameter Analysis """

""" 4. Model Evaluation """

""" 5. Model Testing """

