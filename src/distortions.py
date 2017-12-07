import math
import numpy as np


def shift(data_set, count):
    if count < 0:
        return shift_left(data_set, count * -1)
    else:
        return shift_right(data_set, count)


def crop_x(data_set, amount):
    return shift(shift(shift(data_set, amount), -2 * amount), amount)


def shift_right(data_set, count=1):
    # Each row is an object.

    dim = int(math.sqrt( len(data_set[0]) ))

    for i in range (0, len(data_set)):
        row = data_set[i]
        row = np.insert(row, 0, [0] * count)
        for c in range(0, count):
            row = np.delete(row, len(row)-1)

        for y in range(0, dim):
            for x in range(0, count):
                row[x + y*dim] = 0

        data_set[i] = row


    return data_set


def shift_left(data_set, count=1):
    # Each row is an object.

    dim = int(math.sqrt( len(data_set[0]) ))

    for i in range (0, len(data_set)):
        row = data_set[i]

        row = np.append(row, [0] * count)

        for c in range(0, count):
            row = np.delete(row, 0)

        for y in range(0, dim):
            for x in range(0, count):
                row[(dim - x - 1) + y*dim] = 0

        data_set[i] = row


    return data_set
