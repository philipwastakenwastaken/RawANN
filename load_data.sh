# create data folder
mkdir -p data/
cd data/

# download data
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz > t10k-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz > t10k-labels-idx1-ubyte.gz

# unzip data
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
