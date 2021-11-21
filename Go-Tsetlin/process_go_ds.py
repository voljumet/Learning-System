import string
import numpy as np
np.random.seed(400)
import os
import re
import numpy as np


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
            for z in range(2):
                go_flatten_board.append(x+y+str(z))

    # from above loop your go_flatten_board will have:
    # go_flatten_board = ['aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'ba', 'bb', 'bc', 'bd', 'be',...]

    # now we loop through the sgf stoner positions we read from the files,
    # we match each row_id, col_id to its position in the stone bits we initialized above
    for move in stone_positions.split(';'):
        if move != '':
            if move[0] == 'B':  # check if the position for black stone
                first, second = 1, 0
            elif move[0] == 'W':  # check if the position for white stone
                first, second = 0, 1

            ids = move[2:4].strip(']')  # extracts only row_id and columns_id from the string
            if re.match('[a-zA-Z]+', ids):
                bits[go_flatten_board.index(ids + "0")] = first
                bits[go_flatten_board.index(ids + "1")] = second
    return bits  # all bits representations for both players are concatenated to create the training set X


def convert_game_to_bits_2(x_temp, game_size, function_input, moves_to_predict):

    # bits = np.zeros(game_size**2 *2)  # because 9 x 9 is 9**2, and we double that since we will represent each stone or empty intersection with 2 bits
    game_size_alphabets = alphabet[:game_size]  # if game size is 9, it has 9 elements
    remove_last = ''
    if function_input == "black":
        remove_last = "W"
    elif function_input == "white":
        remove_last = "B"

    go_flatten_board = []
    for y in game_size_alphabets:
        for x in game_size_alphabets:
            for z in range(2):
                go_flatten_board.append(y + x + str(z))

    go_flatten_board_small = []
    for y in game_size_alphabets:
        for x in game_size_alphabets:
            go_flatten_board_small.append(y + x)

    x_array, y_array = [], []

    if x_temp[:-1][0] == remove_last:
        x_temp.pop()

    if moves_to_predict == 0:
        x_array, y_array = recursive(x_temp, go_flatten_board, go_flatten_board_small, game_size, x_array, y_array)
    else:
        x_array, y_array = predict_second_move(x_temp, go_flatten_board, go_flatten_board_small, game_size, function_input, moves_to_predict)

    return x_array, y_array

def predict_second_move(x_temp, go_flatten_board, go_flatten_board_small, game_size, function_input, moves_to_predict):
    bits = np.zeros(game_size ** 2 * 2)

    if function_input == "black":
        moves_to_predict -= 1

    for move in x_temp[:moves_to_predict]:
        if move[0] == 'B':  # check if the position for black stone
            first, second = 1, 0
        elif move[0] == 'W':  # check if the position for white stone
            first, second = 0, 1

        ids = move[2:4].strip(']')  # extracts only row_id and columns_id from the string
        if re.match('[a-zA-Z]+', ids):
            bits[go_flatten_board.index(ids + "0")] = first
            bits[go_flatten_board.index(ids + "1")] = second

    label = go_flatten_board_small.index(x_temp[moves_to_predict][2:4])

    return bits, label

def recursive(x_temp, go_flatten_board, go_flatten_board_small, game_size, x_result=None, y_result=None):
    bits = np.zeros(game_size ** 2 * 2)

    if y_result is None:  # create a new result if no intermediate was given
        x_result, y_result = [], []

    if len(x_temp) != 2:    # stop the recursion if there are no more moves
        for move in x_temp[:-1]:
            if move[0] == 'B':  # check if the position for black stone
                first, second = 1, 0
            elif move[0] == 'W':  # check if the position for white stone
                first, second = 0, 1

            ids = move[2:4].strip(']')  # extracts only row_id and columns_id from the string
            if re.match('[a-zA-Z]+', ids):
                bits[go_flatten_board.index(ids + "0")] = first
                bits[go_flatten_board.index(ids + "1")] = second

        label = go_flatten_board_small.index(x_temp[-1:][0][2:4])

        x_result.append(bits)
        y_result.append(label)
        recursive(x_temp[:-1], go_flatten_board, go_flatten_board_small, game_size, x_result, y_result)

    return x_result, y_result


def create_TM_representations(dataset_dir, game_size, function_input, moves_to_predict):
    ''' we loop through all sgf files we have and read each file (game sample stones positions with winner label) '''
    X = []
    Y = []
    win = 0
    loss = 0
    draw = 0
    for direct, _, files in os.walk(dataset_dir):
        for file_ in files:
            if file_.endswith(".sgf"):
                sgf_file = open(direct + '/' + file_, 'r')

                # we have 4 elements in each loaded list from each sgf sample file. We are interested in the first and last only
                sample = sgf_file.readlines()
                if re.findall('(RE\[W)', sample[0]) != []:  # we use regular expression to check if results in favor of white. Class 0  for white winning
                    label = 0
                    loss += 1
                elif re.findall('(RE\[B)', sample[0]) != []:  # we use regular expression to check if results in favor of black. class 1 for white losing
                    label = 1
                    win += 1
                else:
                    label = 2  # we use regular expression to check if no results reported so it is a draw
                    draw += 1

                # we get the move information in terms of row id and column id, we then convert those to bits representations using a helper function
                if function_input == "win":
                    stone_bits_representation = convert_game_to_bits(sample[3], game_size)
                    X.append(stone_bits_representation)
                    Y.append(label)
                else:
                    x_temp = []
                    for p in sample[3].split(';'):
                        if p != '':
                            ids = p[2:4].strip(']')
                            if re.match('[a-zA-Z]+', ids):
                                x_temp.append(p)

                    if len(x_temp) > 2:
                        if moves_to_predict == 0:
                            stone_bits_representation, label = convert_game_to_bits_2(x_temp, game_size, function_input,
                                                                                      moves_to_predict)
                            for i, each in enumerate(stone_bits_representation):
                                X.append(each)
                                Y.append(label[i])
                        else:
                            if moves_to_predict < len(x_temp) != 2:  # stop the recursion if there are no more moves
                                stone_bits_representation, label = convert_game_to_bits_2(x_temp, game_size, function_input, moves_to_predict)
                                X.append(stone_bits_representation)
                                Y.append(label)

    print("Games with win: ", win)
    print("Games with loss: ", loss)
    print("Games with draw: ", draw)
    print(function_input+" next moves: ", len(Y))

    return np.array(X), np.array(Y)

alphabet = list(string.ascii_lowercase)

