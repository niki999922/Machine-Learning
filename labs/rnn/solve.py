import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import re
import os
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
import re
import re
import numpy as np
import sys



SEQUENCE_LENGTH = 40
EPOCHS = 30
BATCH_SIZE = 64
EPS = 0e-6


# Example input = "testText.txt"
def clear_file(input):
    def isCorrectChar(c):
        return c.isspace() or c == '”' or c == '\'' or (c.isalpha() and (c not in allowed))

    allowed = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ê', 'ê', 'ê']
    inputFile = os.path.join("results", f'{input}.txt')
    outputFile = os.path.join("results", f'___clear___{input}.txt')

    if os.path.exists(outputFile):
        print(f'WARNING: {outputFile} is exist!!!', flush=True)
        return outputFile

    print(f'opening on clearing file {inputFile}', flush=True)
    text = open(inputFile).read()
    text = text.lower()
    text = "".join(list(filter(isCorrectChar, text)))
    text = re.sub('\n+', '\n', re.sub('\n ', '\n', re.sub(' +', ' ', text)))

    print(f'writing cleared file {outputFile}', flush=True)
    open(outputFile, "w").write(text)

    return outputFile

def teach_model(file, teached_model_folder):
    print("Start teaching model", flush=True)
    text = open(file).read()

    chars_int_map = dict((c, i) for i, c in enumerate(sorted(list(set(text)))))
    amount_chars, amount_different_chars = len(text), len(chars_int_map)

    x_arr_dataset_custom_tmp, y_arr_dataset_custom_tmp = [], []

    for i in range(amount_chars - SEQUENCE_LENGTH):
        sequence_from, char_out = text[i:i + SEQUENCE_LENGTH], text[i + SEQUENCE_LENGTH]
        x_arr_tmp = list(map(lambda char: chars_int_map[char], sequence_from))
        x_arr_dataset_custom_tmp.append(x_arr_tmp)
        y_arr_dataset_custom_tmp.append(chars_int_map[char_out])

    x_arr_dataset = numpy.reshape(x_arr_dataset_custom_tmp, (len(x_arr_dataset_custom_tmp), SEQUENCE_LENGTH, 1))
    x_arr_dataset = x_arr_dataset / float(amount_different_chars)
    y_arr_dataset = np_utils.to_categorical(y_arr_dataset_custom_tmp)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model = Sequential()
    model.add(LSTM(256, input_shape=(x_arr_dataset.shape[1], x_arr_dataset.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y_arr_dataset.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    filepath = f'results/{teached_model_folder}/' + "epoch_{epoch:02d}__loss_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(x_arr_dataset, y_arr_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    print("End teaching model", flush=True)

def generate_text_RNN(file, teached_model_file_path, amount_sequence=100):
    print(f'Start generating text under teached model on {teached_model_file_path}', flush=True)
    raw_text = open(file).read()

    chars = sorted(list(set(raw_text)))
    chars_int_map = dict((c, i) for i, c in enumerate(chars))
    int_chars_map = dict((i, c) for i, c in enumerate(chars))
    amount_chars, amount_different_chars = len(raw_text), len(chars)

    x_arr_dataset_custom_tmp, y_arr_dataset_custom_tmp = [], []
    for i in range(amount_chars - SEQUENCE_LENGTH):
        sequence_from, char_out = raw_text[i:i + SEQUENCE_LENGTH], raw_text[i + SEQUENCE_LENGTH]
        x_arr_tmp = list(map(lambda char: chars_int_map[char], sequence_from))
        x_arr_dataset_custom_tmp.append(x_arr_tmp)
        y_arr_dataset_custom_tmp.append(chars_int_map[char_out])

    x_arr_dataset = numpy.reshape(x_arr_dataset_custom_tmp, (len(x_arr_dataset_custom_tmp), SEQUENCE_LENGTH, 1))
    x_arr_dataset = x_arr_dataset / float(amount_different_chars)
    y_arr_dataset = np_utils.to_categorical(y_arr_dataset_custom_tmp)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model = Sequential()
    model.add(LSTM(256, input_shape=(x_arr_dataset.shape[1], x_arr_dataset.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y_arr_dataset.shape[1], activation='softmax'))
    model.load_weights(teached_model_file_path)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    start_sequence_id = numpy.random.randint(0, len(x_arr_dataset_custom_tmp) - 1)
    start_sequence = x_arr_dataset_custom_tmp[start_sequence_id]
    sequence_from = "".join([int_chars_map[value] for value in start_sequence])
    print(f'Start phrase:\n{sequence_from}', flush=True)
    print("Generating:", flush=True)
    for i in range(amount_sequence):
        x = numpy.reshape(start_sequence, (1, len(start_sequence), 1))
        x = x / float(amount_different_chars)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_chars_map[index]
        sys.stdout.write(result)
        start_sequence.append(index)
        start_sequence = start_sequence[1:len(start_sequence)]
    print(f'\nEnd generating text', flush=True)


def get_windows(file, n, k, m):
    text = open(file).read()
    print(f'Start generating text for {file}', flush=True)
    print(f'n = {n}', flush=True)
    print(f'k = {k}', flush=True)
    print(f'm = {m}', flush=True)
    w = set()
    for i in range(len(text) - n + 1):
        w.add(text[i:i + n])
    print("Number of windows: ", len(w))

    w_to_int = dict((c, i) for i, c in enumerate(w))
    int_to_w = dict((i, c) for i, c in enumerate(w))
    matrix = [[0 for _ in range(len(w))] for _ in range(len(w))]
    for i in range(len(text) - n):
        cur_w = text[i:i + n]
        next_w = text[(i + 1):(i + n + 1)]
        matrix[w_to_int[cur_w]][w_to_int[next_w]] += 1
    matrix = np.array(norm_matrix(matrix))
    # print(np.array(matrix))

    lines = text.split('\n')
    start = np.random.randint(0, len(lines) - 1)
    if len(lines[start]) < k:
        print(f'So big {k} for "{lines[start]}"')
    elif k < n:
        print(f'Must k > m"')
    else:
        # prefix = lines[start][0:k]
        prefix = "А за окном"
        print(f'Start:\n{prefix}')
        print("Gen:")
        start_window = prefix[len(prefix) - n:]
        for i in range(m):
            sug_next_pos = get_all_by_max(matrix[w_to_int[start_window]])
            if len(sug_next_pos) == 0:
                print(f'End after {i} symbols')
                break
            elif len(sug_next_pos) == 1:
                ind = 0
            else:
                ind = np.random.randint(0, len(sug_next_pos) - 1)
            next_pos = sug_next_pos[ind]
            # print(next_pos)
            start_window = int_to_w[next_pos]
            sys.stdout.write(start_window[len(start_window) - 1])
    print('End generating text', flush=True)


def get_all_by_max(data):
    max_v = max(data)
    res = []
    for i in range(len(data)):
        if abs(data[i] - max_v) < EPS:
            res.append(i)
    return res


def norm_matrix(matrix):
    new_matrix = []
    for row in matrix:
        sum_v = sum(row)
        if sum_v != 0:
            new_row = list(map(lambda x: x / sum_v, row))
            new_matrix.append(new_row)
        else:
            new_matrix.append(row)
    return new_matrix


if __name__ == '__main__':
    # raw_file = "testText"
    # teached_model_folder = "testFolder"
    # teached_model = "epoch_30__loss_3.1055.hdf5"
    raw_file = "evgeny_onegin"
    teached_model_folder = "evgeny"
    teached_model = "epoch_30__loss_2.4439.hdf5"
    teached_model_file_path = f'results/{teached_model_folder}/{teached_model}'

    cleared_file = clear_file(raw_file)
    # teach_model(cleared_file, teached_model_folder)

    # generate_text_RNN(cleared_file, teached_model_file_path, amount_sequence=120)

    get_windows(cleared_file, 5, 10, 200)


