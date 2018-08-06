import numpy as np
import random
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from pickle import load

print("ME BEAN")
X = np.load("input/X.npy")
Y = np.load("input/Y.npy")

print(X[10].shape)
print(len(X))