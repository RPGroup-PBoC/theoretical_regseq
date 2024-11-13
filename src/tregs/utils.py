import numpy as np

def smoothing(y, windowsize=3):
    """
    Applies a moving average smoothing to a 1D array.

    Args:
        y (np.array): The 1D array to smooth.
        windowsize (int, optional): The size of the smoothing window. Must be an odd integer. Defaults to 3.

    Returns:
        np.array: Smoothed array, with length reduced by `windowsize - 1`.

    Raises:
        RuntimeError: If `windowsize` is not an odd integer.
    """

    if windowsize % 2 != 1:
        raise RuntimeError("Window size has to be odd.")
        
    cut = int((windowsize - 1) / 2)
    out_vec = np.zeros(len(y) - 2 * cut)
    for i in range(cut, len(y) - cut):
        out_vec[i-cut] = np.sum(y[(i-cut):(i+cut+1)])/windowsize
        
    return out_vec

def smoothing_2d(y, windowsize=3):
    """
    Applies a moving average smoothing to each row of a 2D array.

    Args:
        y (np.array): The 2D array to smooth, with shape (rows, columns).
        windowsize (int, optional): The size of the smoothing window. Must be an odd integer. Defaults to 3.

    Returns:
        np.array: Smoothed 2D array, with the same number of rows and reduced columns by `windowsize - 1`.

    Raises:
        RuntimeError: If `windowsize` is not an odd integer.
    """
    if windowsize % 2 != 1:
        raise RuntimeError("Window size has to be odd.")
    
    cut = int((windowsize - 1) / 2)
    num_rows = y.shape[0]
    num_cols = y.shape[1]
    out = np.zeros((num_rows, num_cols - 2 * cut))
    
    for row in range(num_rows):
        for i in range(cut, num_cols - cut):
            out[row, i - cut] = np.sum(y[row, (i - cut):(i + cut + 1)]) / windowsize
            
    return out