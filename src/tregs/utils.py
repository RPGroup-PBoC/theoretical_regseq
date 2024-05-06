import numpy as np

def smoothing(y, windowsize=3):

    if windowsize % 2 != 1:
        raise RuntimeError("Window size has to be odd.")
        
    cut = int((windowsize - 1) / 2)
    out_vec = np.zeros(len(y) - 2 * cut)
    for i in range(cut, len(y) - cut):
        out_vec[i-cut] = np.sum(y[(i-cut):(i+cut+1)])/windowsize
        
    return out_vec

def smoothing_2d(y, windowsize=3):
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