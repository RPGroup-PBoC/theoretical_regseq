import numpy as np



def match_seqs(mut_list, wtseq):

    wtlist = np.array(list(wtseq))
    seqlen = len(wtseq)
    all_mutarr = np.zeros((len(mut_list), seqlen))

    for i, mut in enumerate(mut_list):
        s = np.array(list(mut))
        all_mutarr[i, :seqlen] = (wtlist != s)
    
    return all_mutarr