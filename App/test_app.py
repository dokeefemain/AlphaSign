import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
encoding = pd.read_csv("lib/datasets/key.csv")
model = tf.keras.models.load_model("lib/Models/GermanModel/german")


height = 50
width = 50


path = "lib/datasets/test1.jpg"

image = Image.open(path)
image = image.resize((width,height))
image = np.asarray(image)

test_image = image.astype('float64')

test_image -= np.mean(test_image, axis = 0)
image = test_image.reshape((1,50,50,3))
ind = np.argmax(model.predict(image))

print(encoding[encoding["Encoding"] == ind]["Label"].values)