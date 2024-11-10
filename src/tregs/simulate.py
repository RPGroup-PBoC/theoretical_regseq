import math
import numpy as np
import pandas as pd
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


def generate_emap(seq, fixed=False,
                  fixed_value=1,
                  max_mut_energy=0.5):
    
    '''
    Generates an energy matrix for a binding site.
    Mutations can either have fixed energy values or randomly assigned energies.

    Args:
        seq (str): The reference binding site sequence for which the energy matrix is generated.
        fixed (bool): If True, uses a fixed value for mutations; otherwise, values are random.
        fixed_value (float): The energy value used for all mutations if fixed is True.
        max_mut_energy (float): The maximum energy for mutations if fixed is False.

    Returns:
        np.ndarray: The energy matrix for the sequence.
    '''

    nt_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    emat = np.zeros((4, len(seq)))
    for i,nt in enumerate(seq):
        for j in range(4):
            if j != nt_index[nt]:
                if fixed:
                    emat[j, i] = fixed_value
                else:
                    emat[j, i] = np.random.uniform(low=0.1, high=max_mut_energy)
    
    return emat


def get_dna_cnt(n_seqs):

    '''
    Generates an array of DNA copy numbers for a given number of sequences. Each copy number is drawn from an exponential distribution, scaled, and then rounded up to the nearest integer.

    Args:
        n_seqs (int): The number of DNA sequences for which to generate copy numbers.

    Returns:
        np.array: An array containing the rounded-up copy numbers for each DNA sequence.
    '''

    dna_cnt = np.random.exponential(1, size=n_seqs) * 10

    dna_cnt_up = []
    for cnt in dna_cnt:
        dna_cnt_up.append(math.ceil(cnt))

    return np.array(dna_cnt_up)


## computing pbound and fold change using canonical ensemble

def constitutive_pbound(p_seq, n_NS, n_p, p_emat, ep_wt=0):

    '''
    Calculate the probability of RNAP being bound for a constitutively transcribed promoter.
    
    Args:
        p_seq (str): Sequence of RNAP binding site.
        n_NS (float): Number of non-specific binding sites.
        n_p (float): Copy number of RNAP.
        p_emat (np.ndarray): Energy matrix for the RNAP.
        ep_wt (float, optional): Binding energy of the RNAP at the wild-type binding site. Defaults to 0.

    Returns:
        float: Probability of RNAP being bound.
    '''

    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)

    pbound = (n_p / n_NS * w_p) / (1 + n_p / n_NS * w_p)

    return pbound


def simrep_pbound(p_seq, r_seq, n_NS, n_p, n_r,
                  p_emat, r_emat, ep_wt, er_wt):
    
    '''
    Calculate the probability of RNAP being bound for a promoter with simple repression regulatory architecture.

    Args:
        p_seq (str): Sequence of the RNAP binding site.
        r_seq (str): Sequence of the repressor binding site.
        n_NS (float): Number of non-specific binding sites.
        n_p (float): Copy number of RNAP.
        n_r (float): Copy number of repressor.
        p_emat (np.ndarray): Energy matrix for the RNAP.
        r_emat (np.ndarray): Energy matrix for the repressor.
        ep_wt (float): Binding energy of RNAP at the wild-type binding site.
        er_wt (float): Binding energy of repressor at the wild-type binding site.

    Returns:
        float: Probability of RNAP being bound in the presence of a repressor.
    '''
     
    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)
    w_r = get_weight(r_seq, r_emat, e_wt=er_wt)

    z = np.zeros(3)
    z[0] = 1
    z[1] = n_p / n_NS * w_p
    z[2] = n_r / n_NS * w_r

    return z[1] / np.sum(z)


def simrep_pbound_cp(p_seq, r_seq, p_emat, r_emat, P, R, M, N,
                     ep_wt, er_wt, ep_NS, er_NS):
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
        ep_wt (float): binding energy to the wild type RNAP binding
            site. Defaults to 0.
        er_wt (float): binding energy to the wild type repressor binding
            site. Defaults to 0.
        ep_NS (float): RNAP binding affinity at the non-specific
            binding sites (in kBT units). Defaults to 0.
        er_NS (float): repressor binding affinity at the non-specific
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


def simact_pbound(p_seq, a_seq, n_NS, n_p, n_a, p_emat, a_emat,
                  ep_wt, ea_wt, e_int_pa):
    '''
    Calculate the probability of RNAP being bound for a promoter with simple activation regulatory architecture.

    Args:
        p_seq (str): Sequence of the RNAP binding site.
        a_seq (str): Sequence of the activator binding site.
        n_NS (float): Number of non-specific binding sites.
        n_p (float): Copy number of RNAP.
        n_a (float): Copy number of activator.
        p_emat (np.ndarray): Energy matrix for RNAP.
        a_emat (np.ndarray): Energy matrix for activator.
        ep_wt (float): Energy of RNAP at the wild-type binding site.
        ea_wt (float): Energy of activator at the wild-type binding site.
        e_int_pa (float): Interaction energy between RNAP and activator.

    Returns:
        float: Probability of RNAP being bound.
    '''

    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)
    w_a = get_weight(a_seq, a_emat, e_wt=ea_wt)

    z = np.zeros(4)
    z[0] = 1
    z[1] = (n_p / n_NS) * w_p
    z[2] = (n_a / n_NS) * w_a
    z[3] = z[1] * z[2] * np.exp(-e_int_pa)

    return (z[1] + z[3]) / np.sum(z)


def doublerep_pbound(p_seq, r1_seq, r2_seq, n_NS, n_p, n_r1, n_r2,
                     p_emat, r1_emat, r2_emat, 
                     ep_wt, er1_wt, er2_wt, e_int_r1r2,
                     gate='AND'):
    '''
    Calculate the probability of RNAP being bound for a promoter regulated by two repressors, configured either in an AND or OR logic gate.

    Args:
        p_seq (str): Sequence of the RNAP binding site.
        r1_seq (str), r2_seq (str): Sequences of the two repressor binding sites.
        n_NS (float): Number of non-specific binding sites.
        n_p (float): Copy number of RNAP.
        n_r1 (float), n_r2 (float): Copy numbers of the two repressors.
        p_emat (np.ndarray): Energy matrix for RNAP.
        r1_emat (np.ndarray), r2_emat (np.ndarray): Energy matrices for the repressors.
        ep_wt (float): Energy of RNAP at the wild-type binding site.
        er1_wt (float), er2_wt (float): Energies of the wild-type repressor binding sites.
        e_int_r1r2 (float): Interaction energy between the two repressors.
        gate (str, optional): Logic gate configuration, 'AND' or 'OR'. Defaults to 'AND'.

    Returns:
        float: Probability of RNAP being bound considering the configuration of two repressors.
    '''

    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)
    w_r1 = get_weight(r1_seq, r1_emat, e_wt=er1_wt)
    w_r2 = get_weight(r2_seq, r2_emat, e_wt=er2_wt)

    # OR gate: only need one repressor to be present for repression to occur,
    # i.e. no expression if either repressor is bound
    if gate == 'OR':
        z = np.zeros(5)
        z[0] = 1
        z[1] = n_p / n_NS * w_p
        z[2] = n_r1 / n_NS * w_r1
        z[3] = n_r2 / n_NS * w_r2
        z[4] = (n_r1 / n_NS * w_r1) * (n_r2 / n_NS * w_r2) * np.exp(-e_int_r1r2)
        pbound = z[1] / np.sum(z)

    # AND gate: need both repressors to be present for repression to occur
    elif gate == 'AND':
        z = np.zeros(7)
        z[0] = 1
        z[1] = n_p / n_NS * w_p
        z[2] = n_r1 / n_NS * w_r1
        z[3] = n_r2 / n_NS * w_r2
        z[4] = (n_p / n_NS * w_p) * (n_r1 / n_NS * w_r1)
        z[5] = (n_p / n_NS * w_p) * (n_r2 / n_NS * w_r2)
        z[6] = (n_r1 / n_NS * w_r1) * (n_r2 / n_NS * w_r2) * np.exp(-e_int_r1r2)
        pbound = (z[1] + z[4] + z[5]) / np.sum(z)

    return pbound


def doubleact_pbound(p_seq, a1_seq, a2_seq, n_NS, n_p, n_a1, n_a2,
                     p_emat, a1_emat, a2_emat, 
                     ep_wt, ea1_wt, ea2_wt, e_int_a1a2, e_int_pa1, e_int_pa2, 
                     gate='AND'):
    '''
    Calculate the probability of RNAP being bound for a promoter regulated by two activators, configured either in an AND or OR logic gate.

    Args:
        p_seq (str): Sequence of the RNAP binding site.
        a1_seq (str), a2_seq (str): Sequences of the two activator binding sites.
        n_NS (float): Number of non-specific binding sites.
        n_p (float): Copy number of RNAP.
        n_a1 (float), n_a2 (float): Copy numbers of the two activators.
        p_emat (np.ndarray): Energy matrix for RNAP.
        a1_emat (np.ndarray), a2_emat (np.ndarray): Energy matrices for the activators.
        ep_wt (float): Energy of RNAP at the wild-type binding site.
        ea1_wt (float), ea2_wt (float): Energies of the wild-type activator binding sites.
        e_int_a1a2 (float), e_int_pa1 (float), e_int_pa2 (float): Interaction energies between activators and between each activator and RNAP.
        gate (str, optional): Logic gate configuration, 'AND' or 'OR'. Defaults to 'AND'.

    Returns:
        float: Probability of RNAP being bound considering the configuration of two activators.
    '''

    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)
    w_a1 = get_weight(a1_seq, a1_emat, e_wt=ea1_wt)
    w_a2 = get_weight(a2_seq, a2_emat, e_wt=ea2_wt)

    if gate == 'AND':
        z = np.zeros(8)
        z[0] = 1
        z[1] = n_a1 / n_NS * w_a1
        z[2] = n_a2 / n_NS * w_a2
        z[3] = (n_a1 / n_NS * w_a1) * (n_a2 / n_NS * w_a2) * np.exp(-e_int_a1a2)
        z[4] = n_p / n_NS * w_p
        z[5] = (n_p / n_NS * w_p) * (n_a1 / n_NS * w_a1) * np.exp(-e_int_pa1)
        z[6] = (n_p / n_NS * w_p) * (n_a2 / n_NS * w_a2) * np.exp(-e_int_pa2)
        z[7] = z[5] * (n_a2 / n_NS * w_a2)  * np.exp(-e_int_pa2) * np.exp(-e_int_a1a2)
        pbound = np.sum(z[4:]) / np.sum(z)

    elif gate == 'OR':
        z = np.zeros(8)
        z[0] = 1
        z[1] = n_a1 / n_NS * w_a1
        z[2] = n_a2 / n_NS * w_a2
        z[3] = (n_a1 / n_NS * w_a1) * (n_a2 / n_NS * w_a2)
        z[4] = n_p / n_NS * w_p
        z[5] = (n_p / n_NS * w_p) * (n_a1 / n_NS * w_a1) * np.exp(-e_int_pa1)
        z[6] = (n_p / n_NS * w_p) * (n_a2 / n_NS * w_a2) * np.exp(-e_int_pa2)
        z[7] = z[5] * (n_a2 / n_NS * w_a2) * np.exp(-e_int_pa2)
        pbound = np.sum(z[4:]) / np.sum(z)

    return pbound


def repact_pbound(p_seq, r_seq, a_seq, n_NS, n_p, n_r, n_a,
                  p_emat, r_emat, a_emat,
                  ep_wt, er_wt, ea_wt, e_int_pa):
    
    '''
    Calculate the probability of RNAP being bound for a promoter regulated by both a repressor and an activator.

    Args:
        p_seq (str): Sequence of the RNAP binding site.
        r_seq (str): Sequence of the repressor binding site.
        a_seq (str): Sequence of the activator binding site.
        n_NS (float): Number of non-specific binding sites.
        n_p (float): Copy number of RNAP.
        n_r (float): Copy number of repressor.
        n_a (float): Copy number of activator.
        p_emat (np.ndarray): Energy matrix for RNAP.
        r_emat (np.ndarray): Energy matrix for repressor.
        a_emat (np.ndarray): Energy matrix for activator.
        ep_wt (float): Energy of RNAP at the wild-type binding site.
        er_wt (float): Energy of repressor at the wild-type binding site.
        ea_wt (float): Energy of activator at the wild-type binding site.
        e_int_pa (float): Interaction energy between RNAP and activator.

    Returns:
        float: Probability of RNAP being bound considering the effects of both repressor and activator.
    '''

    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)
    w_r = get_weight(r_seq, r_emat, e_wt=er_wt)
    w_a = get_weight(a_seq, a_emat, e_wt=ea_wt)

    z = np.zeros(5)
    z[0] = 1
    z[1] = n_p / n_NS * w_p
    z[2] = n_r / n_NS * w_r
    z[3] = n_a / n_NS * w_a
    z[4] = (n_p / n_NS * w_p) * (n_a / n_NS * w_a) * np.exp(-e_int_pa)
    
    return (z[1] + z[4]) / np.sum(z)


## building synthetic datasets

def sim_helper(mutants, func_pbound, regions, *args):

    """
    Helper function to simulate binding probabilities for given mutants and binding regions.
    
    This function iterates over a list of mutant sequences, applies a provided probability function to 
    subsets of these sequences defined by specified regions, and collects the results in a DataFrame.

    Args:
        mutants (list): List of mutant DNA sequences to simulate.
        func_pbound (function): Function to calculate binding probabilities.
        regions (list of tuples): List of tuples where each tuple contains start and end indices 
                                  defining regions in the mutants to apply func_pbound.
        *args: Additional arguments to pass to func_pbound.

    Returns:
        pd.DataFrame: A DataFrame with columns for mutant sequences and their calculated binding probabilities.
    """

    l_tr = []
    for mut in mutants:
        rv = {}
        rv['seq'] = mut.upper()

        mut_seqs = []
        for region in regions:

            assert region[1] <= len(mut), f"Region {region} out of bounds for sequence of length {len(mut)}"

            seq = mut[region[0]:region[1]].upper()
            mut_seqs.append(seq)

        rv['pbound'] = func_pbound(*mut_seqs, *args)

        l_tr.append(rv)

    return pd.DataFrame.from_records(l_tr)


def biased_mutants(promoter_seq,
                   num_mutants=10000,
                   mutrate=0.1,
                   allowed_alph=None):
    """
    Generate mutants of a given promoter sequence with a bias towards non-AT nucleotides.
    
    This function selectively mutates bases in the promoter sequence that are not 'A' or 'T', 
    aiming to introduce mutations at a specified rate into these specific bases.

    Args:
        promoter_seq (str): The original promoter DNA sequence to mutate.
        num_mutants (int): The number of unique mutants to generate.
        mutrate (float): The mutation rate applied specifically to non-AT bases.
        allowed_alph (list): List of allowed nucleotides for mutation.

    Returns:
        list: A list of mutant sequences generated from the original sequence.
    """

    indices = []
    subseq = ''
    for i, nt in enumerate(promoter_seq):
        if nt not in ['A', 'T']:
            indices.append(i)
            subseq += nt

    if not subseq:  # Prevent division by zero
        return [promoter_seq]

    _mutants = np.unique(mutations_rand(subseq,
                                        rate=mutrate * len(promoter_seq) / len(subseq),
                                        num_mutants=num_mutants,
                                        allowed_alph=allowed_alph,
                                        number_fixed=True,
                                        keep_wildtype=True))

    mutants = []
    for mutant in _mutants:
        mutseq = ''
        mut_index = 0
        for j in range(len(promoter_seq)):
            if j not in indices:
                mutseq += promoter_seq[j]
            else:
                mutseq += mutant[mut_index]
                mut_index += 1
        mutants.append(mutseq)
    
    return mutants


def sim(promoter_seq, func_pbound, binding_site_seqs, *args,
        preset_mutants=None,
        num_mutants=5000,
        mutrate=0.1,
        biased=False,
        allowed_alph=None,
        scaling_factor=100):
    """
    Simulate the binding probabilities for a given promoter sequence using predefined or generated mutants.

    This function allows simulation with either a provided list of mutant sequences or by generating
    mutants based on the given parameters. It calculates the binding probabilities for specified 
    regions within these sequences.

    Args:
        promoter_seq (str): The promoter sequence for simulation.
        func_pbound (function): Function to calculate binding probabilities.
        binding_site_seqs (list): Sequences defining specific binding sites within the promoter.
        preset_mutants (list, optional): Predefined list of mutant sequences to use.
        num_mutants (int, optional): Number of mutants to generate if not using preset.
        mutrate (float, optional): Mutation rate for generating mutants.
        biased (bool, optional): Whether to apply biased mutation generation.
        allowed_alph (list, optional): Allowed nucleotides for mutations.
        scaling_factor (int, optional): Factor to scale the calculated probabilities.

    Returns:
        pd.DataFrame: DataFrame containing the sequences, original counts, mutated counts, and normalized counts.
    """
    
    if preset_mutants is None:
        if biased:
            mutants = biased_mutants(promoter_seq,
                                    mutrate=mutrate,
                                    num_mutants=num_mutants,
                                    allowed_alph=allowed_alph)
        else:
            mutants = np.unique(mutations_rand(promoter_seq,
                                        rate=mutrate,
                                        num_mutants=num_mutants,
                                        allowed_alph=allowed_alph,
                                        number_fixed=True,
                                        keep_wildtype=True))
    else:
        mutants = preset_mutants
    
    regions = []
    for bss in binding_site_seqs:
        start, end = find_binding_site(promoter_seq, bss)
        regions.append((start, end))

    df_sim = sim_helper(mutants, func_pbound, regions, *args)
    
    dna_cnt = get_dna_cnt(len(df_sim))
    df_sim['ct_0'] = dna_cnt
    df_sim = df_sim[df_sim.ct_0 != 0.0]

    df_sim['ct_1'] = 0.1 + df_sim['ct_0'] * df_sim['pbound'] * scaling_factor
    df_sim['ct_1'] = df_sim['ct_1'].astype(int)

    df_sim['ct_0'] = df_sim['ct_0'].astype(float)
    df_sim['ct_1'] = df_sim['ct_1'].astype(float)
    df_sim['norm_ct_1'] = df_sim['ct_1'] / df_sim['ct_0']

    return df_sim