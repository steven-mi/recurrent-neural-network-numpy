import numpy as np
from numpy import argmax


# this class converts a string/one hot encoded vector
# into a one hot encoded vector/strings
class Translator:

    def __init__(self, characters):
        self.characters = characters
        self.characters_size = len(characters)

        # define a mapping of chars to integers
        self.char_to_int = dict((c, i) for i, c in enumerate(characters))
        self.int_to_char = dict((i, c) for i, c in enumerate(characters))

    def to_integer(self, words):
        integer_encoded = [self.char_to_int[char] for char in words]
        return integer_encoded

    def to_one_hot(self, words):
        # integer encode input data
        integer_encoded = [self.char_to_int[char] for char in words]
        one_hot = []
        for value in integer_encoded:
            letter = [0 for _ in range(self.characters_size)]
            letter[value] = 1
            one_hot.append(letter)
        return np.array(one_hot)

    def to_words(self, one_hot):
        inverted = ''

        if len(one_hot.shape) == 2:
            for word in one_hot:
                inverted += self.int_to_char[argmax(word)]
        else:
            inverted += self.int_to_char[argmax(one_hot)]
        return inverted
