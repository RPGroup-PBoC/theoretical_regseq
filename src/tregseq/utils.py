import numpy as np


def flip_boolean(arr):

    nrow, ncol = arr.shape
    flipped_arr = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            if arr[i][j] == 0:
                flipped_arr[i][j] = 1

    return flipped_arr