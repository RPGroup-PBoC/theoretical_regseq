import numpy as np
import pandas as pd

def match_seqs(mut_list, wtseq):
    '''
    given a list of promoter variants, check whether the base identity at each
    position matches the base identity in the wild type sequence.

    Args:
        mut_list (list): list of promoter variant sequences
        wtseq (str): sequence of the wild type promoter

    Returns:
        arr: each row represents a sequence. each column represents a position.
        an entry of 0 means the base identity is wild type. an entry of 1 means
        the base is mutated.
    '''    

    wtlist = np.array(list(wtseq))
    seqlen = len(wtseq)
    all_mutarr = np.zeros((len(mut_list), seqlen))

    for i, mut in enumerate(mut_list):
        s = np.array(list(mut))
        all_mutarr[i, :seqlen] = (wtlist != s)
    
    return all_mutarr


def get_p_b(all_mutarr, n_seqs):
    '''
    compute the probability that each base position is mutated

    Args:
        all_wtarr (arr): boolean array representing whether each base position
            in each promoter variant is mutated
        n_seqs (int): total number of promoter variants

    Returns:
        arr: array of probability distributions of wild type and mutated bases
            at each base position.
    '''    

    tot_mut_cnt = np.sum(all_mutarr, axis=0)
    p_mut = tot_mut_cnt / n_seqs

    return np.asarray([1 - p_mut, p_mut]).T


def get_p_mu(ex_data, nbins, upper_bound):

    bins = np.linspace(0, upper_bound, nbins, dtype=int)
    out = pd.cut(ex_data, bins=bins,
                include_lowest=True, right=False)
    cnt = out.value_counts().tolist()
    cnt.append(len(ex_data) - sum(cnt))
    p_mu = np.asarray(cnt) / sum(cnt)
    return p_mu


def mi_footprint(fpath_to_seqcnt, wtseq):

    df = pd.read_csv(fpath_to_seqcnt)

    all_mutarr = match_seqs(df['seq'].values, wtseq)
    all_wtarr = utils.flip_boolean(all_mutarr)

    seq_cnt = df[['ct_0', 'ct_1']].apply(np.sum).values
    seq_cnt /= np.sum(df['ct'])

    p_mut_tot = count_mut_tot(all_wtarr, all_mutarr, df['ct'].values)

    p_wt_dna = count_mut(all_wtarr, df['ct_0'].values, df['ct'].values)
    p_wt_rna = count_mut(all_wtarr, df['ct_1'].values, df['ct'].values)
    p_mut_dna = count_mut(all_mutarr, df['ct_0'].values, df['ct'].values)
    p_mut_rna = count_mut(all_mutarr, df['ct_1'].values, df['ct'].values)

    info_fp = MI(p_mut_tot, seq_cnt,
                p_wt_dna, p_wt_rna, p_mut_dna, p_mut_rna,
                seqlen=len(wtseq))
    
    return info_fp