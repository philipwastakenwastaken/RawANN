import numpy as np
import data_stream as ds
import nn


np.random.seed(12445512)

epoch_size = 1000

print("Reading data")

# random sample input data
X = ds.read_train_images(epoch_size)

y = ds.read_train_labels(epoch_size)

X_test_images = ds.read_validation_images()
y_test_labels = ds.read_validation_labels()

print("Processing data")
p = []
for col in range(epoch_size):
    entry = (X[:, col].reshape(28 * 28, 1), y[:, col].reshape(10, 1))
    p.append(entry)

validation_data = []
for col in range(10000):
    entry = (X_test_images[:, col].reshape(28 * 28, 1), y_test_labels[:, col].reshape(10, 1))
    validation_data.append(entry)

model = nn.Model([30], 28 * 28, 10)

print("Fitting model")
model.fit(p, 3, 5, 10, validation_data)
