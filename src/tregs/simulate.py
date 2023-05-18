import math
import numpy as np
import pandas as pd
import sympy as sym
from .seq_utils import seq_to_list, find_binding_site
from .wgregseq import mutations_rand
from .solve_utils import solve_biquadratic


def fix_wt(matrix, seq):
    '''
    fix the energy matrix such that the binding energy for the wild type base
    identity at each position is 0

    Args:
        matrix (arr): energy matrix
        seq (str): wild type sequence of binding site

    Returns:
        arr: fixed energy matrix
    '''     

    seq = seq_to_list(seq)
    mat_fixed = []
    for i, row in enumerate(matrix):
        mat_fixed.append([val - row[seq[i]] for val in row])

    return np.asarray(mat_fixed)
    

def get_d_energy(seq, energy_mat, e_wt=0):
    '''
    given an energy matrix and the sequence of the binding site, calculate
    the total binding energy. if the energy matrix is not fixed, we add
    the binding energy to the wild type sequence.

    Args:
        seq (str): sequence of binding site
        energy_mat (arr): energy matrix
        e_wt (int, optional): total binding energy to the wild type binding
        site. Defaults to 0.

    Returns:
        float: total binding energy in kBT units
    '''    

    indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    if energy_mat.shape[0] == 4:
        energy_mat = energy_mat.T
    d_energy = 0
    for i,AA in enumerate(seq):
        d_energy += energy_mat[i][indices[AA]]
        
    return d_energy + e_wt


def get_weight(seq, energy_mat, e_wt=0):
    '''
    compute the Boltzmann weight, exp(- d_energy / kBT). note that here
    d_energy is already in kBT units and so we don't have to perform the division.

    Args:
        seq (str): sequence of binding site
        energy_mat (arr): energy matrix
        e_wt (int, optional): total binding energy to the wild type binding
        site. Defaults to 0.

    Returns:
        float: Boltzmann weight.
    '''    

    d_energy = get_d_energy(seq, energy_mat, e_wt=e_wt)
    
    return np.exp(-d_energy)


## computing pbound and fold change using canonical ensemble

def simrep_pbound(p_seq, r_seq, p_emat, r_emat, n_p, n_r, n_NS,
                  ep_wt=0, er_wt=0):
    '''
    calculate the probability of binding for a gene with the simple repression
    regulatory architecture

    Args:
        p_seq (str): sequence of the RNAP binding site
        r_seq (str): sequence of the repressor binding site
        p_emat (arr): energy matrix for RNAP
        r_emat (arr): energy matrix for the repressor
        n_p (int): number of RNAPs
        n_r (int): number of repressors
        n_NS (int): number of non-specific binding sites
        ep_wt (int, optional): binding energy to the wild type RNAP binding
        site. Defaults to 0.
        er_wt (int, optional): binding energy to the wild type repressor binding
        site. Defaults to 0.

    Returns:
        float: probability of RNAP binding
    '''
    #print(n_p, n_r)

    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)
    w_r = get_weight(r_seq, r_emat, e_wt=er_wt)

    return (n_p / n_NS * w_p) / (1 + n_p / n_NS * w_p + n_r / n_NS * w_r)


def simact_pbound(p_seq, a_seq, p_emat, a_emat, n_p, n_a, n_NS,
                  ep_wt=0, ea_wt=0, e_ap=0):

    p = (n_p / n_NS) * get_weight(p_seq, p_emat, e_wt=ep_wt)
    a = (n_a / n_NS) * get_weight(a_seq, a_emat, e_wt=ea_wt)
    w = np.exp(-e_ap)

    pbound = (p + a * p * w) / (1 + a + p + a * p * w)

    return pbound


def simrep_fc(r_seq, r_emat, n_r, n_NS, e_wt=0):
    '''
    calculate fold change in expression levesl when repressors are introduced
    into the system

    Args:
        r_seq (_type_): sequence of the repressor binding site
        r_emat (_type_): energy matrix for the repressor
        n_r (_type_): number of repressors
        n_NS (_type_): number of non-specific binding sites
        e_wt (int, optional): binding energy to the wild type repressor binding
        site. Defaults to 0.

    Returns:
        float: fold change
    '''    

    w_r = get_weight(r_seq, r_emat, e_wt=e_wt)

    return 1 / (1 + n_r / n_NS * w_r)


def simrep_pbound_cp(p_seq, r_seq, p_emat, r_emat, P, R, M, N,
                     ep_wt=0, er_wt=0, ep_NS=0, er_NS=0):
    '''
    calculate pbound for a promoter with simple repression regulatory
    architecture using the chemical potential approach.

    Args:
        p_seq (str): sequence of the RNAP binding site
        r_seq (str): sequence of the repressor binding site
        p_emat (arr): energy matrix for RNAP
        r_emat (arr): energy matrix for the repressor
        P (int): number of RNA polymerases.
        R (int): number of repressors.
        M (int): number of specific binding sites (i.e. copy number of the promoter).
        N (int): number of non-specific binding sites
        ep_wt (int, optional): binding energy to the wild type RNAP binding
            site. Defaults to 0.
        er_wt (int, optional): binding energy to the wild type repressor binding
            site. Defaults to 0.
        ep_NS (int, optional): RNAP binding affinity at the non-specific
            binding sites (in kBT units). Defaults to 0.
        er_NS (int, optional): repressor binding affinity at the non-specific
            binding sites (in kBT units) Defaults to 0.

    Returns:
        (float): probability of RNAP binding
    '''

    x_p = np.exp(-1 * get_d_energy(p_seq, p_emat, e_wt=ep_wt))
    x_r = np.exp(-1 * get_d_energy(r_seq, r_emat, e_wt=er_wt))
    y_p = np.exp(-1 * ep_NS)
    y_r = np.exp(-1 * er_NS)
    l_p, l_r = solve_biquadratic(P, R, M, N, x_p, y_p, x_r, y_r)

    pbound = l_p * x_p / (1 + l_p * x_p + l_r * x_r)

    #occNS_rnap = l_p * y_p / (1 + l_p * y_p + l_r * y_r)
    #occS_rep = l_r * x_r / (1 + l_p * x_p + l_r * x_r)
    #occNS_rep = l_r * y_r / (1 + l_p * y_p + l_r * y_r)

    return pbound


## performing the simulation

def simrep_helper(mutants, rnap_start, rnap_end, rep_start, rep_end,
                  rnap_emat, rep_emat, n_p, n_r, n_NS,
                  ep_wt, er_wt):

    l_tr = []
    for mut in mutants:
        rv = {}
        rv['seq'] = mut
        rnap_mut = mut[rnap_start:rnap_end].upper()
        rep_mut = mut[rep_start:rep_end].upper()
        rv['pbound'] = simrep_pbound(rnap_mut, rep_mut, rnap_emat, rep_emat,
                                     n_p, n_r, n_NS,
                                     ep_wt=ep_wt, er_wt=er_wt)
        l_tr.append(rv)
    df_simrep = pd.DataFrame.from_records(l_tr)

    return df_simrep


def get_dna_cnt(n_seqs):

    dna_cnt = np.random.exponential(1, size=n_seqs) * 10

    dna_cnt_up = []
    for cnt in dna_cnt:
        dna_cnt_up.append(math.ceil(cnt))

    return dna_cnt


def simrep(wtseq, rnap_wtseq, rep_wtseq, rnap_emat, rep_emat, 
           ep_wt, er_wt, n_NS, n_p, n_r,
           num_mutants=10000, mutrate=0.1, scaling_factor=100):
    
    mutants = np.unique(mutations_rand(wtseq,
                                       rate=mutrate,
                                       num_mutants=num_mutants,
                                       number_fixed=True))

    rnap_start, rnap_end = find_binding_site(wtseq, rnap_wtseq)
    rep_start, rep_end = find_binding_site(wtseq,rep_wtseq)

    df_simrep = simrep_helper(mutants, rnap_start, rnap_end, rep_start, rep_end,
                          rnap_emat, rep_emat, n_p, n_r, n_NS,
                          ep_wt, er_wt)
    
    dna_cnt = get_dna_cnt(len(df_simrep))
    df_simrep['ct_0'] = dna_cnt
    df_simrep = df_simrep[df_simrep.ct_0 != 0.0]

    df_simrep['ct_1'] = 0.1 + df_simrep['ct_0'] * df_simrep['pbound'] * scaling_factor
    df_simrep['ct_1'] = df_simrep['ct_1'].astype(int)
    df_simrep['ct'] = df_simrep['ct_0'] + df_simrep['ct_1']

    df_simrep['ct'] = df_simrep['ct'].astype(float)
    df_simrep['ct_0'] = df_simrep['ct_0'].astype(float)
    df_simrep['ct_1'] = df_simrep['ct_1'].astype(float)
    df_simrep['norm_ct_1'] = df_simrep['ct_1'] / df_simrep['ct_0']

    return df_simrep