import sys
import numpy as np
from utils import *
import random
from pickle import load

def sample(parameters, char_to_ix):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []

    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "indices". We'll stop if we reach 20 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 20):
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.matmul(Wax, x) + np.matmul(Waa, a_prev) + b)
        z = np.matmul(Wya, a) + by
        y = softmax(z)

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(np.arange(vocab_size), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        counter += 1

    if (counter == 20):
        indices.append(char_to_ix['\n'])

    return indices

# Get number of names to print and gender from command line parameters
args = sys.argv
num_samples = 5
if(args[1].isdigit()):
    num_samples = int(args[1])
parameters = None
if(args[2].lower() == 'm'):
    parameters = load(open('output/parameters_male.pkl', 'rb'))
else:
    parameters = load(open('output/parameters_female.pkl', 'rb'))

# Load dictionaries for vocabulary
char_to_ix = load(open('output/char_to_ix.pkl', 'rb'))
ix_to_char = load(open('output/ix_to_char.pkl', 'rb'))

# Generate num_samples names
for i in range(0, num_samples):
    sampled_indices = sample(parameters, char_to_ix)
    print_sample(sampled_indices, ix_to_char)
