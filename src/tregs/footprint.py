import numpy as np
import pandas as pd
from .utils import smoothing, smoothing_2d
from .mpl_pboc import plotting_style

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 12})
plotting_style()


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


def get_p_b(all_mutarr, n_seqs, pseudocount=0):
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
    tot_mut_cnt = np.sum(all_mutarr, axis=0)
    p_mut = (tot_mut_cnt + pseudocount) / (n_seqs + pseudocount)

    return np.asarray([1 - p_mut, p_mut]).T


def bin_expression_levels(mu_data, nbins, upper_bound):
    """
    Bins the expression levels data into specified number of bins with a given upper bound.

    Args:
        mu_data (np.array or pd.Series): Array or series containing expression level data.
        nbins (int): The number of bins to divide the data into.
        upper_bound (float): The maximum value for the bins.

    Returns:
        tuple:
            - np.array: Array of bin indices for each element in the input data.
            - np.array: Array of counts of elements in each bin.

    Raises:
        ValueError: If `nbins` is less than 1 or `upper_bound` is less than the maximum of `mu_data`.
    """
    
    bins = np.linspace(0, upper_bound, nbins).tolist()
    bins.append(int(max(mu_data) + 1))

    binned = pd.cut(mu_data, bins=bins,
                    labels=np.arange(nbins),
                    include_lowest=True, right=False)
    #binned = pd.qcut(mu_data, 2,
    #                labels=np.arange(nbins))
    
    mu_bins = binned.values
    bin_cnt = binned.value_counts(sort=False).values

    return mu_bins, np.asarray(bin_cnt)

'''
def bin_expression_levels2(mu_data, nbins):

        df_tmp = pd.DataFrame(mu_data)
        df_tmp.sort_values(by='norm_ct_1', ascending=True, inplace = True)

        splits = np.array_split(df_tmp, nbins)
        for i in range(len(splits)):
            splits[i]['group'] = i + 1
        df_tmp = pd.concat(splits)
        df_tmp.sort_index(inplace=True)
        mu_bins = df_tmp['group'].values - 1

        bin_cnt = df_tmp.groupby(['group']).size()

        return mu_bins, np.asarray(bin_cnt)
'''


def get_p_mu(bin_cnt, n_seqs):

    """
    Calculate the probability of sequences falling into each bin.

    Args:
        bin_cnt (np.array): Array containing counts of sequences in each bin.
        n_seqs (int): Total number of sequences that were binned.

    Returns:
        np.array: An array of probabilities for each bin.
    """

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
    """
    Calculates the mutual information (MI) between mutations and expression levels 
    across each position of a promoter sequence.

    Args:
        list_p_b (list of np.array): List of arrays where each entry corresponds to 
                                     the probabilities of the wild type and mutated base
                                     at each position of the promoter.
        p_mu (np.array): Array of probabilities for each expression level bin.
        list_joint_p (list of np.array): List of 3D arrays where each entry contains 
                                         joint probabilities of base identity and 
                                         expression bins at each base position.
        len_promoter (int, optional): The total number of bases in the promoter 
                                      sequence. Defaults to 160.

    Returns:
        list: Mutual information values for each position along the promoter, measured in bits.

    Note:
        This function assumes that the input probabilities are valid and that the logarithmic 
        operations do not encounter log(0) due to proper handling of zero probabilities with 
        pseudocounts.
    """

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
                       smoothed=True, windowsize=3):
    """
    Calculates the information footprint of a promoter.

    Args:
        mut_list (list of str): List of mutant sequences.
        mu_data (np.array): Expression levels corresponding to each sequence in mut_list.
        wtseq (str): Wild type sequence of the promoter.
        nbins (int): Number of bins to divide the expression data into.
        upper_bound (float): The maximum value to consider for the bins in expression data.
        pseudocount (float, optional): Pseudocount to avoid division by zero in probability calculations.
        len_promoter (int, optional): Length of the promoter analyzed. Defaults to 160.
        smoothed (bool, optional): Whether to smooth the resulting footprint. Defaults to True.
        windowsize (int, optional): Size of the window used for smoothing. Defaults to 3.

    Returns:
        np.array: Array of mutual information values across the promoter positions.
    """

    n_seqs = len(mut_list)

    #print('start time: {}'.format(datetime.datetime.now()))
    all_mutarr = match_seqs(wtseq, mut_list)
    #print('finished match_seqs: {}'.format(datetime.datetime.now()))
    list_p_b = get_p_b(all_mutarr, n_seqs, pseudocount=pseudocount)
    #print('finished calculating p_b: {}'.format(datetime.datetime.now()))
    mu_bins, bin_cnt = bin_expression_levels(mu_data, nbins, upper_bound)
    p_mu = get_p_mu(bin_cnt, n_seqs)
    #print('finished calculating p_mu: {}'.format(datetime.datetime.now()))
    list_joint_p = get_joint_p(all_mutarr, mu_bins, nbins,
                               pseudocount=pseudocount, len_promoter=len_promoter)
    #print('finished calculating joint probability distribution: {}'.format(datetime.datetime.now()))

    footprint = MI(list_p_b, p_mu, list_joint_p, len_promoter=len_promoter)
    #print('finished calculating mutual information: {}'.format(datetime.datetime.now()))
    if smoothed:
        footprint = smoothing(footprint, windowsize=windowsize)

    return footprint


def get_expression_shift(mut_list, mu_data, wtseq,
                         len_promoter=160):
    """
    Calculates the expression shift due to mutations at each position of a promoter.

    Args:
        mut_list (list of str): List of mutant sequences.
        mu_data (np.array): Expression levels corresponding to each mutant in mut_list.
        wtseq (str): Wild type sequence of the promoter.
        len_promoter (int, optional): Length of the promoter analyzed. Defaults to 160.

    Returns:
        np.array: Array representing the average expression shift at each promoter position.
    """

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
    
    #if smoothed:
    exshift_list = smoothing(exshift_list, windowsize=3)

    
    return exshift_list


def get_expression_shift_matrix(df, wtseq,
                                len_promoter=160, smoothed=False):
    
    seqs = df['seq'].values
    mu_data = df['norm_ct_1']

    n_seqs = len(seqs)
    avg_mu = np.mean(mu_data)

    def make_int(x):
        dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        return [dict[y] for y in x]

    int_seqs = []
    for i_seq in range(n_seqs):
        seq = seqs[i_seq].upper()
        int_seqs.append(make_int(seq))

    int_wtseq = make_int(wtseq)

    exshift_list = []
    for position in range(len_promoter):
        ex_shift = np.zeros(4)
        for base in range(4):
            if base + 1 != int_wtseq[position]:
                for i_seq in range(n_seqs):
                    int_seq = int_seqs[i_seq]
                    if (int_seq[position] == base + 1):
                        ex_shift[base] += (mu_data[i_seq] - avg_mu) / avg_mu
        ex_shift /= n_seqs
        exshift_list.append(ex_shift)
    
    exshift_arr = np.asarray(exshift_list).T
    
    if smoothed:
        exshift_arr = smoothing_2d(exshift_arr, windowsize=3)
    
    return exshift_arr


## plotting information footprint

def label_binding_site(ax, start, end, max_signal, type, label,
                       lifted=False):
    shade_color = {'P': '#A9BFE3', 'R': '#E8B19D', 'A': '#DCECCB'}
    label_color = {'P': '#738FC1', 'R': '#D56C55', 'A': '#7AA974'}
    ax.axvspan(start, end, alpha=0.7, color=shade_color[type])
    if lifted:
        y_coord = max_signal * 1.37
        text_y_coord = max_signal * 1.42
    else:
        y_coord = max_signal * 1.15
        text_y_coord = max_signal * 1.2
    ax.add_patch(mpl.patches.Rectangle((start, y_coord),
                                       end-start,
                                       max_signal * 0.2,
                                       facecolor=label_color[type],
                                       clip_on=False,
                                       linewidth = 0))
    ax.text(start + 0.5 * (end-start), text_y_coord, label, fontsize = 12, color = 'k',
            ha='center', va='baseline')
    

def plot_footprint(promoter, df, region_params,
                   nbins=2, up_scaling_factor=1,
                   pseudocount=10**(-6),
                   smoothed=True, windowsize=3,
                   max_signal=None, x_lims=None,
                   fig_width=10, fig_height=2.9,
                   outfile=None,
                   return_fp=False):
    
    mut_list = df['seq'].values
    mu_data = df['norm_ct_1']
    upper_bound = up_scaling_factor * np.mean(mu_data)

    len_promoter = len(promoter)

    footprint = get_info_footprint(mut_list, mu_data, promoter, nbins, upper_bound,
                                               pseudocount=pseudocount,
                                               smoothed=smoothed, windowsize=windowsize,
                                               len_promoter=len_promoter)
    exshift_list = get_expression_shift(mut_list, mu_data.values, promoter,
                                        len_promoter=len_promoter)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if x_lims is not None:
        left_x_lim = x_lims[0]
        right_x_lim = x_lims[1]
    else:
        left_x_lim = -115
        right_x_lim = 45

    ax.set_xlim(left_x_lim, right_x_lim)

    if max_signal is None:
        max_signal = max(footprint)
    
    ax.set_ylim(top=max_signal*1.15)
    for region in region_params:
        if len(region)==4:
            label_binding_site(ax, region[0], region[1], max_signal, region[2], region[3])
        else:
            label_binding_site(ax, region[0], region[1], max_signal, region[2], region[3],
                           lifted=region[4])

    if smoothed:
        cut = int((windowsize - 1) / 2)
        x = np.arange(left_x_lim + cut, right_x_lim - cut)
    else:
        x = np.arange(left_x_lim, right_x_lim)
    shiftcolors = [('#D56C55' if exshift > 0 else '#738FC1') for exshift in exshift_list]
    ax.bar(x, footprint, color=shiftcolors, edgecolor=None, linewidth=0)
    ax.set_ylabel('Information (bits)', fontsize=16)
    ax.set_xlabel('Position relative to TSS', fontsize=16)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()

    if return_fp:
        return footprint
    

def plot_exshift(promoter, df,
                 vmax_scaling=0.5,
                 outfile=None):
    exshift = get_expression_shift_matrix(df, promoter)

    fig, ax = plt.subplots(figsize=(15, 2))

    vmax = np.max(np.abs(exshift)) * vmax_scaling
    vmin = -vmax

    div_colors = sns.diverging_palette(258, 16, s=56, l=50, n=15, sep=5, center='light', as_cmap=True)
    hm = sns.heatmap(exshift, cmap=div_colors, ax=ax, vmin=vmin, vmax=vmax, center=0)
    ax.set_yticklabels(['A', 'C', 'G', 'T'], rotation=360, fontsize=14)
    ax.set_xlabel('Position Relative to TSS', fontsize=16)

    tick_positions = np.arange(15, 161, 20)
    tick_labels = np.arange(-100, 41, 20)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=360, fontsize=15)

    cbar = hm.collections[0].colorbar
    cbar.ax.set_position([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar.set_label('Expression\nshift', rotation=90, labelpad=15, fontsize=14)
    cbar.ax.tick_params(labelsize=13) 

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        plt.show()