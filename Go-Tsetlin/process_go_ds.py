import sys
import os
from os import path
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from sklearn.metrics import f1_score
from os.path import isfile, join
import string
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from tm import MultiClassTsetlinMachine
from os import listdir
from time import time
import re
import numpy as np
import random
np.random.seed(400)
import pickle
import nltk
import json
import collections
import pandas as pd
import seaborn as sns
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



def convert_game_to_bits(stone_positions, game_size):
    '''takes stone positions string for black and white players and convert it to bits.
    when we flatten the board of size 9x9 for example, we get 81 bits,
    we represent black stones with 1 0 when found on a position, if white stone: 0 1, otherwise, we leave it 0 0'''

    bits = np.zeros(game_size**2 *2)  # because 9 x 9 is 9**2, and we double that since we will represent each stone or empty intersection with 2 bits
    # print(bits)
    game_size_alphapets = alphabet[:game_size]  # if game size is 9, it has 9 elements
    # we create a flatten version of the Go game when 9X9 by combining all possible positions and make a flat list from.
    # you can extend this if worked with 13X13 or 19X19. Please add more alphapet in this case to the array above
    # the below flatten array will be used to easily convert sgf row and col ids to the bits array associated indices
    go_flatten_board = []
    for x in game_size_alphapets:
        for y in game_size_alphapets:
            go_flatten_board.append(x+y)

    # from above loop your go_flatten_board will have: ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'ba', 'bb', 'bc', 'bd', 'be',...]

    # now we loop through the sgf stoner positions we read from the files,
    # we match each row_id, col_id to its position in the stone bits we initialized above
    for p in stone_positions.split(';'):
        if p != '':
            if p[0] == 'B':  # check if the position for black stone
                ids = p[2:4].strip(']')  # extracts only row_id and columns_id from the string
                if re.match('[a-zA-Z]+', ids):
                    row = ids[0]  # get rowid
                    col = ids[1]  # get col id

                    # now we use the created go_flatten_board to convert to bits for the black stone player representations
                    bits[go_flatten_board.index(row + col)] = 1
                    bits[go_flatten_board.index(row + col)+1] = 0
            elif p[0] == 'W':  # check if the position for white stone
                ids = p[2:4]  # extracts only row_id and columns_id from the string
                if re.match('[a-zA-Z]+', ids):
                    row = ids[0]  # get rowid
                    col = ids[1]  # get col id

                    # now we use the created go_flatten_board to convert to bits for the white stone player representations
                    bits[go_flatten_board.index(row + col)] = 0
                    bits[go_flatten_board.index(row + col)+1] = 1
    return bits  # all bits representations for both players are concatenated to create the training set X


def create_TM_representations(dataset_dir, game_size):
    ''' we loop through all sgf files we have and read each file (game sample stones positions with winner label) '''

    X = []
    Y = []
    for direct, _, files in os.walk(dataset_dir):
        for file_ in files:
            if file_.endswith(".sgf"):
                sgf_file = open(direct + '/' + file_, 'r')

                # we have 4 elements in each loaded list from each sgf sample file. We are interested in the first and last only
                sample = sgf_file.readlines()
                if re.findall('(RE\[W)', sample[0]) != []:  # we use regular expression to check if results in favor of white. Class 0  for white winning
                    label = 0
                elif re.findall('(RE\[B)', sample[0]) != []:  # we use regular expression to check if results in favor of black. class 1 for white losing
                    label = 1
                else:
                    label = 2  # we use regular expression to check if no results reported so it is a draw

                # we get the move information in terms of row id and column id, we then convert those to bits representations using a helper function
                stone_bits_representation = convert_game_to_bits(sample[3], game_size)

                X.append(stone_bits_representation)
                Y.append(label)

    return np.array(X), np.array(Y)

alphabet = list(string.ascii_lowercase)

# game_size = 9  # determines number of columns and rows in the game, this must match with the dataset we are loading
# sgf_dir = 'go9'  # here you speciy your Go dataset sgf files directory
# X, Y = create_TM_representations(sgf_dir, game_size)  # where X is all samples represented in bits, Y is all the labels

# print(X)



