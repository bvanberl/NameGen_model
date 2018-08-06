import numpy as np
from keras import backend as K
from keras.models import load_model, Model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from pickle import load
import tensorflow as tf

# Define layers
V_size = 28
n_a = 300
reshapor = Reshape((1, V_size))
#LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(V_size, activation='softmax')

# Define dictionaries for vocabulary reference
ix_to_char = load(open('output/ix_to_char.pkl', 'rb'))
char_to_ix = load(open('output/char_to_ix.pkl', 'rb'))


# Returns the one hot encoding of a vector
def one_hot(x):
    x = tf.multinomial(x, 1)
    #x = np.random.choice(a = V_size, p = x.ravel())
    x = tf.one_hot(x, V_size)
    #x = RepeatVector(1)(x)
    return x


# define the function for the inference model
def inference_model_1(LSTM_cell, densor, vocab_size, n_a):
    # define the inputs to the model
    x0 = Input(shape=(1, vocab_size))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    # empty list to store the outputs
    outputs = []

    out = []
    # loop over the characters in the target name (max sequence length)
    for i in range(0,15):
    #while (K.argmax(out) != char_to_ix['\n']):
        # pass the input to the LSTM cell(trained previoulsy, as it is a global layer), with the initial hidden states.
        # Then re-initialize the hidden states for the next time step with the output states of the current time step
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        # output of the current time step
        out = densor(a)

        # append to list
        outputs.append(out)

        # call the one_hot() function previously defined, with the 'softmax' activated dense layer (shape = m, vocab_size) as input
        x = Lambda(one_hot)(out)

    # create the model instance
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


def predict_and_sample(inference_model, x_initialiser, a_initialiser, c_initialiser):
    # predict the output sequence
    pred = inference_model.predict([x_initialiser, a_initialiser, c_initialiser])

    # find the indexes of characters that have the greatest probability in each time steps output
    indices = np.argmax(pred, axis=-1)

    return indices


def generate_name():
    model = load_model('output/model.h5')
    LSTM_cell = model.layers[6]

    inference_model = inference_model_1(LSTM_cell, densor, V_size, n_a)

    # initialise variables to pass as input
    x_initialiser = np.zeros((1, 1, V_size))
    a_initialiser = np.zeros((1, n_a))
    c_initialiser = np.zeros((1, n_a))

    # Initialize RNN with 1 random character
    random_ix = np.random.randint(2, len(char_to_ix))
    print(ix_to_char[random_ix])
    x_initialiser[0, 0, random_ix] = 1

    indices = predict_and_sample(inference_model, x_initialiser, a_initialiser, c_initialiser)

    # so that the shape changes from (dino_name_length, num_examples) to (num_examples, dino_name_length)
    indices = indices.swapaxes(0, 1)

    # join the characters in the output to form a legible name
    name = ''.join([ix_to_char[i] for i in indices[0]])

    print("Generated name = " + str(name))


generate_name()