import tensorflow as tf
from tensorflow import keras

import numpy as np

#Step 1 - Getting training and testing data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

#Step 2 - Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Step 2.2 - Compile the network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""Loss function: gives probability of each number
Optimizer: method to train the network. There are lots of optimizer"""

#Step 3 - Train
model.fit(train_images, train_labels, epochs=5)
"""Each epoch is one "sweep" on the training data. 
Aim to not "overtune" with too many epochs."""

#Step 4 - Evaluate
model.evaluate(test_images, test_labels)

#Step 5 - Predict
scores = model.predict(test_images[0:1])
print(np.argmax(scores))

#Step 6 - Save the model
#model.save("my_model.h5", True, True)
#del model #deletes existing model
"""Can't save this model because it has no input_shape"""