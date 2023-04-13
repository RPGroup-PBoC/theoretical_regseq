import numpy as np
import pandas as pd
from .utils import smoothing

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

    #tot_mut_cnt = np.sum(all_mutarr * mu_data.values[:, np.newaxis], axis=0)
    #p_mut = tot_mut_cnt / np.sum(mu_data)
    tot_mut_cnt = np.sum(all_mutarr, axis=0)
    p_mut = tot_mut_cnt / n_seqs

    return np.asarray([1 - p_mut, p_mut]).T


def bin_expression_levels(mu_data, nbins, upper_bound):
    bins = np.linspace(0, upper_bound, nbins, dtype=int).tolist()
    bins.append(int(max(mu_data) + 1))

    binned = pd.cut(mu_data, bins=bins,
                    labels=np.arange(nbins),
                    include_lowest=True, right=False)
    
    mu_bins = binned.values
    bin_cnt = binned.value_counts().values
    return mu_bins, bin_cnt


def get_p_mu(bin_cnt, n_seqs):
    return bin_cnt / n_seqs


def get_joint_p(all_mutarr, mu_bins, nbins, n_seqs,
                pseudocount=10**(-6), len_promoter=160):
    list_joint_p = []
    for position in range(len_promoter):
        joint_p = np.zeros((2, nbins)) + pseudocount
        # adding a pseudocount to prevent zero-division error
        for i in range(n_seqs):
            for j in range(nbins):
                if (all_mutarr[i][position] == 0) & (mu_bins[i] == j):
                    joint_p[0][j] += 1
                elif (all_mutarr[i][position] == 1) & (mu_bins[i] == j):
                    joint_p[1][j] += 1

        joint_p /= np.sum(joint_p)
        list_joint_p.append(joint_p)
    return list_joint_p


def MI(list_p_b, p_mu, list_joint_p,
       len_promoter=160):
    mutual_info = []
    for position in range(len_promoter):
        p_b = list_p_b[position]
        joint_p = list_joint_p[position]

        mi = 0
        for i in range(len(p_mu)):
            mi += joint_p[0][i] * np.log2(joint_p[0][i] / (p_b[0] * p_mu[i]))
            mi += joint_p[1][i] * np.log2(joint_p[1][i] / (p_b[1] * p_mu[i]))
        mutual_info.append(mi)
    return mutual_info


def get_info_footprint(mut_list, mu_data, wtseq,
                       nbins, upper_bound,
                       pseudocount=10**(-6), len_promoter=160):
    n_seqs = len(mut_list)

    all_mutarr = match_seqs(mut_list, wtseq)
    list_p_b = get_p_b(all_mutarr, n_seqs)
    mu_bins, bin_cnt = bin_expression_levels(mu_data, nbins, upper_bound)
    p_mu = get_p_mu(bin_cnt, n_seqs)
    list_joint_p = get_joint_p(all_mutarr, mu_bins, nbins, n_seqs,
                               pseudocount=pseudocount, len_promoter=len_promoter)
    footprint = MI(list_p_b, p_mu, list_joint_p)
    return footprint


def get_expression_shift(mut_list, mu_data, wtseq,
                         len_promoter=160, smoothed=True, windowsize=1):
    n_seqs = len(mu_data)
    avg_mu = np.mean(mu_data)
    all_mutarr = match_seqs(mut_list, wtseq)

    exshift_list = []
    for position in range(len_promoter):
        ex_shift = 0
        for i_seq in range(n_seqs):
            ex_shift += all_mutarr[i_seq][position] * (1 - mu_data[i_seq] / avg_mu)
        ex_shift /= n_seqs
        exshift_list.append(ex_shift)
    
    if smoothed:
        exshift_list = smoothing(exshift_list, windowsize=windowsize)
    
    return exshift_list


## code to calculate information footprint based on the eLife paper

def flip_boolean(arr):

    nrow, ncol = arr.shape
    flipped_arr = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            if arr[i][j] == 0:
                flipped_arr[i][j] = 1

    return flipped_arr


def count_mut(all_mutarr, cnt_seq, tot_cnt_seq):

    _cnt = np.multiply(all_mutarr, cnt_seq[:, np.newaxis])
    cnt = np.sum(_cnt, axis=0) / np.sum(tot_cnt_seq)
    return cnt


def count_tot(all_wtarr, all_mutarr, cnt_seq):
    wt_cnt = np.multiply(all_wtarr, np.asarray(cnt_seq)[:, np.newaxis])
    mut_cnt = np.multiply(all_mutarr, np.asarray(cnt_seq)[:, np.newaxis])
    tot_wt_cnt = np.sum(wt_cnt, axis=0)
    tot_mut_cnt = np.sum(mut_cnt, axis=0)
    _cnt = np.stack([tot_wt_cnt, tot_mut_cnt])
    cnt = _cnt / np.sum(_cnt, axis=0)
    return cnt


def MI_old(p_mut_tot, seq_cnt,
       p_wt_dna, p_wt_rna, p_mut_dna, p_mut_rna,
       seqlen=160):

    mutual_info = []
    for i in range(seqlen):
        mi = p_wt_dna[i] * np.log2(p_wt_dna[i] / (p_mut_tot[:, i][0] * seq_cnt[0]))
        mi += p_wt_rna[i] * np.log2(p_wt_rna[i] / (p_mut_tot[:, i][0] * seq_cnt[1]))
        mi += p_mut_dna[i] * np.log2(p_mut_dna[i] / (p_mut_tot[:, i][1] * seq_cnt[0]))
        mi += p_mut_rna[i] * np.log2(p_mut_rna[i] / (p_mut_tot[:, i][1] * seq_cnt[1]))
        mutual_info.append(mi)
        
    return mutual_info


def footprint_old(df, wtseq):

    all_mutarr = match_seqs(df['seq'].values, wtseq)
    all_wtarr = flip_boolean(all_mutarr)

    seq_cnt = df[['ct_0', 'ct_1']].apply(np.sum).values
    seq_cnt /= np.sum(df['ct'])

    p_mut_tot = count_tot(all_wtarr, all_mutarr, df['ct'].values)

    p_wt_dna = count_mut(all_wtarr, df['ct_0'].values, df['ct'].values)
    p_wt_rna = count_mut(all_wtarr, df['ct_1'].values, df['ct'].values)
    p_mut_dna = count_mut(all_mutarr, df['ct_0'].values, df['ct'].values)
    p_mut_rna = count_mut(all_mutarr, df['ct_1'].values, df['ct'].values)

    info_fp = MI_old(p_mut_tot, seq_cnt,
                p_wt_dna, p_wt_rna, p_mut_dna, p_mut_rna,
                seqlen=len(wtseq))
    
    return info_fp