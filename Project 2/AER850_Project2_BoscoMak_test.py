import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# directories
test_dir = 'Data/train'

# load data

test = tf.keras.utils.image_dataset_from_directory(
  directory = test_dir,
  image_size=(500, 500),
  )

class_names = test.class_names

#load images
img1 = tf.keras.utils.load_img(
    "Data/test/crack/test_crack.jpg", target_size=(500, 500)
)
img2 = tf.keras.utils.load_img(
    "Data/test/missing-head/test_missinghead.jpg", target_size=(500, 500)
)
img3 = tf.keras.utils.load_img(
    "Data/test/paint-off/test_paintoff.jpg", target_size=(500, 500)
)

# model load

model = keras.models.load_model("model.keras")

# normalization

img1_norm = tf.keras.utils.img_to_array(img1)
img2_norm = tf.keras.utils.img_to_array(img2)
img3_norm = tf.keras.utils.img_to_array(img3)

img1_norm = tf.expand_dims(img1_norm, 0)
img2_norm = tf.expand_dims(img2_norm, 0)
img3_norm = tf.expand_dims(img3_norm, 0)

img1_norm = img1_norm/255.0
img2_norm = img2_norm/255.0
img3_norm = img3_norm/255.0

# prediction

pred1 = model.predict(img1_norm)
score1 = tf.nn.softmax(pred1[0])

pred2 = model.predict(img2_norm)
score2 = tf.nn.softmax(pred2[0])

pred3 = model.predict(img3_norm)
score3 = tf.nn.softmax(pred3[0])

# plot

plt.figure(1)
plt.imshow(img1)
plt.axis('off')
plt.title("True Crack Classification Label: crack\nPredicted Crack Classification Label: " + class_names[np.argmax(score1)])
plt.text(15, 40, s = "Crack: " + str(100*np.round(np.max(score1[0]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 70, s = "Missing Head: " + str(100*np.round(np.max(score1[1]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 100, s = "Paint Off: " + str(100*np.round(np.max(score1[2]), 5)) + "%", size = 'x-large', c = 'green')

plt.figure(2)
plt.imshow(img2)
plt.axis('off')
plt.title("True Crack Classification Label: missing-head\nPredicted Crack Classification Label: " + class_names[np.argmax(score2)])
plt.text(15, 40, s = "Crack: " + str(100*np.round(np.max(score2[0]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 70, s = "Missing Head: " + str(100*np.round(np.max(score2[1]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 100, s = "Paint Off: " + str(100*np.round(np.max(score2[2]), 5)) + "%", size = 'x-large', c = 'green')

plt.figure(3)
plt.imshow(img3)
plt.axis('off')
plt.title("True Crack Classification Label: paint-off\nPredicted Crack Classification Label: " + class_names[np.argmax(score3)])
plt.text(15, 40, s = "Crack: " + str(100*np.round(np.max(score3[0]), 4)) + "%", size = 'x-large', c = 'green')
plt.text(15, 70, s = "Missing Head: " + str(100*np.round(np.max(score3[1]), 5)) + "%", size = 'x-large', c = 'green')
plt.text(15, 100, s = "Paint Off: " + str(100*np.round(np.max(score3[2]), 4)) + "%", size = 'x-large', c = 'green')