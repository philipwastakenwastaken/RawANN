import numpy as np


def read_validation_images(size=0):
    data = []

    train_images = open("data/t10k-images-idx3-ubyte", "rb")
    train_images.seek(4)
    images = int.from_bytes(train_images.read(4), byteorder='big')
    rows = int.from_bytes(train_images.read(4), byteorder='big')
    columns = int.from_bytes(train_images.read(4), byteorder='big')

    if (size == 0):
        size = images

    for k in range(size):
        img = []

        for i in range(rows * columns):
            x = int.from_bytes(train_images.read(1), byteorder='big')
            xf = (x / 255.0)  # convert to [0 ; 1]
            img.append(xf)

        data.append(img)

    train_images.close()
    return np.array(data).T


def read_train_images(size=0):
    data = []

    train_images = open("data/train-images-idx3-ubyte", "rb")
    train_images.seek(4)
    images = int.from_bytes(train_images.read(4), byteorder='big')
    rows = int.from_bytes(train_images.read(4), byteorder='big')
    columns = int.from_bytes(train_images.read(4), byteorder='big')

    if (size == 0):
        size = images

    for k in range(size):
        img = []

        for i in range(rows * columns):
            x = int.from_bytes(train_images.read(1), byteorder='big')
            xf = (x / 255.0)  # convert to [0 ; 1]
            img.append(xf)

        data.append(img)

    train_images.close()
    return np.array(data).T


def read_train_labels(size=0):
    data = []

    train_labels = open("data/train-labels-idx1-ubyte", "rb")
    train_labels.seek(4)
    labels = int.from_bytes(train_labels.read(4), byteorder='big')

    if (size == 0):
        size = labels

    for _ in range(size):
        label = int.from_bytes(train_labels.read(1), byteorder='big')
        label_vector = [0.0] * 10
        label_vector[label] = 1
        data.append(label_vector)

    train_labels.close()
    return np.array(data).T


def read_validation_labels(size=0):
    data = []

    train_labels = open("data/t10k-labels-idx1-ubyte", "rb")
    train_labels.seek(4)
    labels = int.from_bytes(train_labels.read(4), byteorder='big')

    if (size == 0):
        size = labels

    for _ in range(size):
        label = int.from_bytes(train_labels.read(1), byteorder='big')
        label_vector = [0.0] * 10
        label_vector[label] = 1
        data.append(label_vector)

    train_labels.close()
    return np.array(data).T
