import pandas
from pandas import read_csv
import numpy as np
from tensorflow.python.keras.utils import to_categorical


MIN_LENGTH = 5
MAX_LENGTH = 5


def load_from_file(folder, x_filename, y_filename, times_filename):
    x = read_csv(folder + "/" + x_filename)
    y = read_csv(folder + "/" + y_filename)
    times = pandas.read_csv(folder + "/" + times_filename)
    return x.values, y.values.flatten().astype(int), times.values.flatten()


def generate_samples(X, Y):
    x = []
    y = []
    for n in range(len(X) - MAX_LENGTH):
        x.append([])
        for i in range(MAX_LENGTH):
            x[n].append(X[n + i])
        y.append(Y[n + MAX_LENGTH - 1])

    x = np.array(x)
    y = to_categorical(np.array(y, dtype='int32'))
    return x, y

def _get_system_vitals(_system_vitals, _headers):
    return _system_vitals[_headers]


def get_label_field_name():
    return 'IS_ATTACK'


def _normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())