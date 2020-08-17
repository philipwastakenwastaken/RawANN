import sys
import numpy as np
import data_stream as ds
import nn

np.set_printoptions(threshold=sys.maxsize)
np.random.seed(12445512)

epoch_size = 1024

print("Reading data")

# random sample input data
X = ds.read_train_images(epoch_size)

y = ds.read_train_labels(epoch_size)

print("Processing data")
p = []
for col in range(epoch_size):
    entry = (X[:, col].reshape(28 * 28, 1), y[:, col])
    p.append(entry)

model = nn.Model([16, 16], 28 * 28, 10)

print("Fitting model")
model.fit(p, 0.001, 1, 32)
