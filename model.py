import numpy as np
import random
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from pickle import load
import tensorflow as tf

V_size = 28
n_a = 300  # the number of units for each LSTM cell
X = np.load("input/X.npy")
Y = np.load("input/Y.npy")
print(X.shape)
print(Y.shape)

# Define layers
reshapor = Reshape((1, V_size))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(V_size, activation='softmax')

# Define dictionaries for vocabulary reference
ix_to_char = load(open('output/ix_to_char.pkl', 'rb'))
char_to_ix = load(open('output/char_to_ix.pkl', 'rb'))


# Returns the one hot encoding of a vector
def one_hot(x, vocab_size):
    x = np.random.choice(a = vocab_size, p = x.ravel())
    x = tf.one_hot(x, vocab_size)
    x = RepeatVector(1)(x)
    return x


def bvmodel(Tx, n_a, V_size):
    # Define the inputs to the model
    X = Input(shape=(Tx, V_size))

    # Initial hidden states of the LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    # We will append the output of the LSTM cell at each time step
    outputs = []

    # Iterate over all characters in the input name
    for t in range(Tx):

        # Pick out the one-hot representation of character t, using a Lambda function
        x = Lambda(lambda x: x[:, t, :])(X)

        # Reshape in order to be inputted to the LSTM
        x = reshapor(x)

        # Input x into the LSTM cell and update the hidden states.
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        # Pass output of LSTM into the densor layer, which gives the softmax activation with V_size units
        out = densor(a)

        # Append the output to the outputs list. Now, 'out' is of shape (num_examples, vocab_size).
        # When appending it to the list 'outputs', ultimately the shape becomes (Tx, num_examples, vocab_size) and that is
        # why the actual target outputs have been made to be of shape (dino_name_length, num_examples, vocab_size)
        outputs.append(out)

    # Define the model instance, with the one-hot encoded input names and initial states to the LSTM as input,
    # and one-hot representations of the actual target names as outputs.
    model = Model(inputs=[X, a0, c0], outputs=outputs)

    return model

Tx = 16
model = bvmodel(Tx, 300, V_size)
model.summary()
print("Model defined.")

# Define the optimizer and compile the model
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
2
# initialise variables to feed into the model as initial input
m = X.shape[0] # number of examples
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

# Fit the model. Since the output is a list, we convert Y into a list too
print("Fitting model")
model.fit([X, a0, c0], list(Y), epochs=50)


# Save the model
print("Saving model")
model.save('output/model.h5')

