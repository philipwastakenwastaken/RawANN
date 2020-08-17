import numpy as np
import tensorflow as tf
from tensorflow import keras
import data_stream as ds


epoch_size = 60000
num_classes = 10

print("Reading data")
# random sample input data
X_train = ds.read_train_images(epoch_size).T
y_train = ds.read_train_labels(epoch_size).T

X_test = ds.read_validation_images().T
y_test = ds.read_validation_labels().T

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

inputs = keras.Input(shape=(784,), name="MNIST")
x = keras.layers.Dense(30, activation="relu")(inputs)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
     optimizer=keras.optimizers.Adam(),
     metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=32, epochs=30,
                    validation_split=0.2)

test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss", test_scores[0])
print("Test accuracy", test_scores[1])
