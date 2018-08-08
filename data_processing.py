import os.path
import numpy as np
from pickle import dump
import tensorflow as tf
from keras.utils import to_categorical


# Returns the one hot encoding of a vector
def one_hot(x, vocab_size):
    x = tf.one_hot(x, vocab_size)
    return x


def prepare_input(file_name):
    if os.path.isfile(file_name):
        names = open(file_name, 'r').read()
    else:
        return []

    V = list(set(names)) # Get vocabulary of characters
    V.append('.')
    V = sorted(V)
    V_size = len(V)
    print('There are %d total characters and %d unique characters in the data.' % (len(names), len(V)))

    # Create and store dictionaries for conversion between characters and vocabulary indices.
    char_to_ix = {ch: i for i, ch in enumerate(sorted(V))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(V))}
    #dump(char_to_ix, open('output/char_to_ix.pkl', 'wb'))
    #dump(ix_to_char, open('output/ix_to_char.pkl', 'wb'))

    # Get maximum name length in the data set. Add 1 for None at beginning of x and \n at end of y
    names = names.split("\n")
    max_length = len(max(names, key=len)) + 1
    print("V = " + str(V))
    print("max_length = " + str(max_length))

    # Initialize X and Y.
    X = np.zeros((len(names), max_length, V_size))
    Y = np.zeros((X.shape))

    for i in range(0, len(names) - 1):
        X[i] = to_categorical([char_to_ix["."]] + [char_to_ix[ch] for ch in names[i]] + [char_to_ix["\n"] for i in range(1, max_length - len(names[i]))], V_size)
        Y[i] = to_categorical([char_to_ix[ch] for ch in names[i]] + [char_to_ix["\n"] for i in range(0, max_length - len(names[i]))], V_size)
        if i%10000 == 0:
            print(i)
    Y = np.transpose(Y, (1, 0, 2))
    np.save('input/X.npy', X)
    np.save('input/Y.npy', Y)

    return X, Y, len(V)


prepare_input('input/male_names.txt')
