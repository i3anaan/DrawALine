import math
import numpy as np
from visualise_images import visualize_img


def extend_dataset_grow(data_X, data_y, count=1):
    output_X = data_X.copy()
    output_y = data_y

    for i in range(0, count, 1):
        output_X = shrink(grow(output_X))
        output_y = output_y
    output_X = np.concatenate((data_X, output_X))
    output_y = np.concatenate((data_y, output_y))

    return (output_X, output_y)


def extend_dataset_shift(data_X, data_y):
    output_X = data_X
    output_y = data_y

    output_X = np.concatenate((output_X, shift(data_X, 1, 1)))
    output_y = np.concatenate((output_y, data_y))
    output_X = np.concatenate((output_X, shift(data_X, -1, 1)))
    output_y = np.concatenate((output_y, data_y))
    output_X = np.concatenate((output_X, shift(data_X, 1, -1)))
    output_y = np.concatenate((output_y, data_y))
    output_X = np.concatenate((output_X, shift(data_X, -1, -1)))
    output_y = np.concatenate((output_y, data_y))

    return (output_X, output_y)


def shift(data_set, x, y):
    output = data_set
    print(visualize_img(output.reshape((28,28))))
    dim = int(math.sqrt(len(data_set[0])))
    if x < 0:
        print("LEFT:")
        output = shift_left(data_set, x * -1)
    else:
        print("RIGHT:")
        output = shift_right(data_set, x)
    if y < 0:
        print("DOWN:")
        #output = shift_down(data_set, y * -1)
    else:
        print("UP:")
        #output = shift_up(data_set, y)
    print(visualize_img(output.reshape((28,28))))
    return output


def crop(data_set, x, y):
    return shift(shift(shift(data_set, x, y), -2 * x, -2 * y), x, y)


def shift_right(data_set, count=1):
    # Each row is an object.
    dim = int(math.sqrt(len(data_set[0])))
    for i in range(0, len(data_set)):
        sample = data_set[i].reshape((28,28))
        for y in range(0, dim):
            row = sample[y]

            row = np.insert(row, 0, [0] * count)
            for c in range(0, count):
                row = np.delete(row, len(row) - 1)

            sample[y] = row
        data_set[i] = sample.reshape(784)

    return data_set


def shift_left(data_set, count=1):
    # Each row is an object.
    dim = int(math.sqrt(len(data_set[0])))
    for i in range(0, len(data_set)):
        sample = data_set[i].reshape((28,28))
        for y in range(0, dim):
            row = sample[y]

            row = np.insert(row, len(row) - 1, [0] * count)
            for c in range(0, count):
                row = np.delete(row, 0)

            sample[y] = row
        data_set[i] = sample.reshape(784)

    return data_set


def shift_up(data_set, count=1):
    dim = int(math.sqrt(len(data_set[0])))

    for i in range(0, len(data_set)):
        row = data_set[i]

        for c in range(0, count * dim):
            row = np.delete(row, 0)
        row = np.append(row, [0] * count * dim)
        data_set[i] = row

    return data_set


def shift_down(data_set, count=1):
    dim = int(math.sqrt(len(data_set[0])))

    for i in range(0, len(data_set)):
        row = data_set[i]

        for c in range(0, count * dim):
            row = np.delete(row, len(row) - 1)
        row = np.insert(row, 0, [0] * count * dim)
        data_set[i] = row

    return data_set


def grow(data_set):
    dim = int(math.sqrt(len(data_set[0])))
    for i in range(0, len(data_set)):
        temp_row = data_set[i].copy()
        for p in range(0, len(temp_row)):
            if (temp_row[p] != 0):
                updated_row = set_cross(data_set, data_set[i], p, dim)
        data_set[i] = updated_row
    return data_set


def shrink(data_set):
    dim = int(math.sqrt(len(data_set[0])))

    for i in range(0, len(data_set)):
        temp_row = data_set[i].copy()
        for p in range(0, len(temp_row)):
            if not check_block(data_set, temp_row, p, dim):
                data_set[i][p] = 0
    return data_set


def set_cross(data_set, row, index, dim):
    row = maybe_set(data_set, row, index + 1, 1)
    row = maybe_set(data_set, row, index - 1, 1)
    row = maybe_set(data_set, row, index - dim, 1)
    row = maybe_set(data_set, row, index + dim, 1)
    return row


def check_block(data_set, row, index, dim):
    return maybe_check(data_set, row, index + 1, 1) and maybe_check(
        data_set, row, index - 1, 1) and maybe_check(
            data_set, row, index - dim, 1) and maybe_check(
                data_set, row, index - dim - 1, 1) and maybe_check(
                    data_set, row, index - dim + 1, 1) and maybe_check(
                        data_set, row, index + dim, 1) and maybe_check(
                            data_set, row, index + dim - 1, 1) and maybe_check(
                                data_set, row, index + dim + 1, 1)


def maybe_set(data_set, row, index, value):
    if (index >= 0 and index < len(row)):
        row[index] = value
    return row


def maybe_check(data_set, row, index, value):
    return index >= 0 and index < len(row) and row[index] == value
