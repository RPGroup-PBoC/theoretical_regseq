import datetime
import numpy as np
import pandas as pd
from .utils import smoothing

def match_seqs(wtseq, mut_list):
    '''
    given a list of promoter variants, check whether the base identity at each
    position matches the base identity in the wild type sequence.

    Args:
        wtseq (str): sequence of the wild type promoter
        mut_list (list): list of promoter variant sequences

    Returns:
        arr: each row represents a sequence. each column represents a position.
        an entry of 0 means the base identity is wild type. an entry of 1 means
        the base is mutated.
    '''    

    wtlist = np.array(list(wtseq))
    wt_arr = np.vstack([wtlist] * len(mut_list))
    mut_arr = np.asarray([list(mutant) for mutant in mut_list.tolist()])
    
    return wt_arr != mut_arr


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
    #bins = np.linspace(0, upper_bound, nbins, dtype=int).tolist()
    bins = np.linspace(0, upper_bound, nbins).tolist()
    bins.append(int(max(mu_data) + 1))

    binned = pd.cut(mu_data, bins=bins,
                    labels=np.arange(nbins),
                    include_lowest=True, right=False)
    
    mu_bins = binned.values
    bin_cnt = binned.value_counts(sort=False).values

    return mu_bins, bin_cnt


def get_p_mu(bin_cnt, n_seqs):
    return bin_cnt / n_seqs


def get_joint_p(all_mutarr, mu_bins, nbins,
                pseudocount=10**(-6), len_promoter=160):
    '''
    calculates the joint probability distribution between mutations and expression
    levels at each base position along the promoter sequence

    Args:
        all_mutarr (bool arr): array indicating whether each position in each
            sequence is mutated (each row represents a sequence. each column
            represents a position. an entry of 0 means the base identity is wild
            type. an entry of 1 means the base is mutated.)
        mu_bins (int arr): array indicating the index of the expression level
            bin each sequence is in.
        nbins (int): total number of bins for expression levels.
        pseudocount (float, optional): Prevents zero division error.
            Defaults to 10**(-6).
        len_promoter (int, optional): length of the promoter. Defaults to 160.

    Returns:
        float arr: a list of the joint probability distributions at each base position
    '''    
    
    mu_bins_tmp = mu_bins[:, np.newaxis]
    list_joint_p = np.zeros((len_promoter, 2, nbins)) + pseudocount
    for b in range(2):
        for mu in range(nbins):
            list_joint_p[:, b, mu] += np.sum(((all_mutarr == b) * (mu_bins_tmp == mu)), axis=0) 
    list_joint_p = list_joint_p / np.sum(list_joint_p, axis=(1, 2))[:, np.newaxis, np.newaxis]
                            
    return list_joint_p


def MI(list_p_b, p_mu, list_joint_p,
       len_promoter=160):
    mutual_info = []
    for position in range(len_promoter):
        p_b = list_p_b[position]
        joint_p = list_joint_p[position]

        mi = 0
        for i in range(len(p_mu)):
            #print(joint_p[0][i], p_b[0] * p_mu[i])
            mi += joint_p[0][i] * np.log2(joint_p[0][i] / (p_b[0] * p_mu[i]))
            #print(joint_p[1][i], p_b[1] * p_mu[i])
            mi += joint_p[1][i] * np.log2(joint_p[1][i] / (p_b[1] * p_mu[i]))
        mutual_info.append(mi)
    return mutual_info


def get_info_footprint(mut_list, mu_data, wtseq,
                       nbins, upper_bound,
                       pseudocount=10**(-6), len_promoter=160,
                       smoothed=True, windowsize=3,
                       fast=False):
    n_seqs = len(mut_list)

    #print('start time: {}'.format(datetime.datetime.now()))
    all_mutarr = match_seqs(wtseq, mut_list)
    #print('finished match_seqs: {}'.format(datetime.datetime.now()))
    list_p_b = get_p_b(all_mutarr, n_seqs)
    #print('finished calculating p_b: {}'.format(datetime.datetime.now()))
    mu_bins, bin_cnt = bin_expression_levels(mu_data, nbins, upper_bound)
    p_mu = get_p_mu(bin_cnt, n_seqs)
    #print('finished calculating p_mu: {}'.format(datetime.datetime.now()))
    list_joint_p = get_joint_p(all_mutarr, mu_bins, nbins,
                               pseudocount=pseudocount, len_promoter=len_promoter)
    #print('finished calculating joint probability distribution: {}'.format(datetime.datetime.now()))
    footprint = MI(list_p_b, p_mu, list_joint_p)
    #print('finished calculating mutual information: {}'.format(datetime.datetime.now()))
    if smoothed:
        footprint = smoothing(footprint, windowsize=windowsize)

    return footprint


def get_expression_shift(mut_list, mu_data, wtseq,
                         len_promoter=160, smoothed=True, windowsize=3):
    n_seqs = len(mu_data)
    avg_mu = np.mean(mu_data)
    all_mutarr = match_seqs(wtseq, mut_list)

    exshift_list = []
    for position in range(len_promoter):
        ex_shift = 0
        for i_seq in range(n_seqs):
            ex_shift += all_mutarr[i_seq][position] * (mu_data[i_seq] / avg_mu - 1)
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

    all_mutarr = match_seqs(wtseq, df['seq'].values)
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