# Import libraries and modules
import numpy as np
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
np.save('./data_dir/X_train.npy', X_train)
np.save('./data_dir/y_train.npy', y_train)
np.save('./data_dir/X_test.npy', X_test)
np.save('./data_dir/y_test.npy', y_test)
