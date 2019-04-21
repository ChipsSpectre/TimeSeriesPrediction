import numpy as np


def generate_sequences(data, in_seq_len, out_seq_len, axis=0):
    """
        Generates sequence patterns for supervised learning along a single axis
        of a multi-dimensional tensor.

        The values along the selected axis are treated as a series of values, 
        which is cutted into overlapping sequence sections.
    """
    patterns = []
    labels = []

    N = data.shape[axis]
    print(N)
    full_seq_len = in_seq_len + out_seq_len
    # first part usable, possible to extract sequences here
    usable_seq_len = N - full_seq_len + 1
    print(usable_seq_len)
    for i in range(usable_seq_len):
        patterns.append(np.take(data, range(i,
                                            i + in_seq_len), axis))
        labels.append(np.take(data, range(i + in_seq_len,
                                          i + full_seq_len), axis))

    return np.array(patterns), np.array(labels)


def train_val_split(data, val_fraction, axis=0):
    """
        Performs train and validation split, according to the desired axis.
        Default axis to split is the first one, since this is format is also 
        used by keras.

        Returns (train, val) as numpy arrays, respectively.
    """
    if val_fraction < 0 or val_fraction > 1:
        raise ValueError("Invalid validation fraction supplied.")

    shape = data.shape
    if axis < 0 or axis >= len(shape):
        raise ValueError("Invalid axis specified.")

    N = shape[axis]

    train_fraction = 1 - val_fraction

    n_train = int(N * train_fraction)
    n_val = N - n_train

    return np.take(data, range(n_train), axis), \
        np.take(data, range(n_train, n_train + n_val), axis)
