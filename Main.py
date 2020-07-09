import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# loading in the digits dataset
digit_data = tf.keras.datasets.mnist

# splitting the data into both testing and training the data
(train_images, train_labels), (test_images, test_labels) = digit_data.load_data()

# creating a list to define each label
class_names = ['Zero', 'One', 'Two', 'Three','Four','Five','Six','Seven', 'Eight', 'Nine']

# shrinking the data down from 0-1 instead of up to 255 (dividing by 255 because they are grey-scale values)
train_images = train_images/255
test_images = test_images/255

# creating a model (sequence of layers)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # flattening the data to pass to the neurons (each image is 28x28 pixels)
    keras.layers.Dense(150, activation="relu"), # hidden dense layer to pass to output layer
    keras.layers.Dense(10, activation="softmax") # output layer of 10 neurons for numbers 0-9
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# training the model on training data
model.fit(train_images, train_labels, epochs=25) # epochs set to 10 to run through all the data 10 times

# saving the model to avoid training it constantly when running
model.save("saved_model")

# loading the saved model
model = keras.models.load_model("saved_model")

# predicting the created model
prediction = model.predict(test_images)

# looping through 5 digits to show both predicted and actual digit values using a grid
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual Digit: " + class_names[test_labels[i]]) # actual digits taken from the class_names list that contains all digit values
    plt.title("Predicted Digit: " + class_names[np.argmax(prediction[i])])# taking the highest number as predicted value using np.argmax
    plt.show()





