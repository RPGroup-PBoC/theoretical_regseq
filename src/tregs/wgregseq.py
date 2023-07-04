## Written by Tom Roeschinger

import numpy as np
import numba


def isint(var):
    """
    Check if variable is of type `int` or type `float`, but an integer.
    """
    
    if not (isinstance(var, int) or isinstance(var, np.int64)):
        if (isinstance(var, float) or isinstance(var, np.float64)):
            if not var.is_integer():
                return False
            else:
                return True
        else:
            return False
    else:
        return True
    

def random_mutation_generator(sequence, rate, num_mutants, number_fixed):
    mutant_list = np.empty(num_mutants, dtype=object)
    for i in range(num_mutants):
        mutant_list[i] = _random_mutation_generator(sequence, rate, number_fixed)
    
    return mutant_list


def mutate_from_index(sequence, index, alph):
    """Generate mutated sequence from wild type sequence given a an index and 
    an alphabet.
    
    Parameters
    ----------
    - sequence : string
        Wild type seqence
    - index : array of tuples
        Each tuple is a position and the mutation at that position.
    - alph : array 
        Alphabet to pick mutation from
        
    Returns
    -------
    - str
        Sequence including mutation
    """
    seq_list = list(sequence)
    for loci, mutation in index:
        seq_list[loci] = filter_mutation(seq_list[loci], alph)[mutation].lower()
    return "".join(seq_list)


def mutate_with_bias(sequence, index, letters, allowed_alph):
    seq_list = list(sequence)
    for _loci in index:
        loci = _loci[0]
        seq_list[loci] = make_mutation(seq_list[loci], letters, allowed_alph).lower()
    return "".join(seq_list)


def mutations_rand(
    sequence, 
    num_mutants,
    rate,
    site_start=0, 
    site_end=None, 
    alph_type="DNA",
    allowed_alph=None,
    number_fixed=False,
    keep_wildtype=False,
):
    """Creates single or double mutants.
    
    
    Parameters
    ----------
    - sequence : string
        DNA sequence that is going to be mutated.
    - num_mutants : int, default None
        Number of mutant sequences. If None, all possible mutatants are created.
    - mut_per_seq : int, default 1
        Number of mutations per sequence.
    - site_start : int, default 0
        Beginning of the site that is about to be mutated.
    - site_end : int, default -1
        End of the site that is about to be mutated.
    - alph_type : string, default "DNA"
        Can either be "DNA" for letter sequences, or "Numeric" for integer sequences.
    - number_fixed : bool
        If True, the number of mutations is fixed as the rate times length of the sequence.
    - keep_wildtype : bool, default False
        If True, adds wild type sequence as first sequence in the list.
        
    Returns
    -------
    - mutants : list
        List of mutant sequences. Each element is a string.
    """
    
    if not (isint(num_mutants) or num_mutants == None):
        raise TypeError("`num_mutants` is of type {} but has to be integer valued.".format(type(num_mutants)))
    
    if not type(rate) == float:
        raise TypeError("`rate` is of type {} but has to be a float.".format(type(rate)))
        
    if not isint(site_start):
        raise TypeError("`site_start` is of type {} but has to be integer valued.".format(type(site_start)))
        
    if not (isint(site_end) or site_end == None):
        raise TypeError("`site_end` is of type {} but has to be integer valued.".format(type(site_end)))
        
    if alph_type not in ["DNA", "Numeric"]:
        raise ValueError("`alph_type` has to be either \"DNA\" or \"Numeric\" ")
        
    
    # Get site to mutate
    
    if site_end==None:
        mutation_window = sequence[site_start:]
    else:
        mutation_window = sequence[site_start:site_end]

    
    # Create list
    if keep_wildtype:
        mutants = np.empty(num_mutants+1, dtype='U{}'.format(len(sequence)))
        mutants[0] = sequence
        i_0 = 1
    else:
        mutants = np.empty(num_mutants, dtype='U{}'.format(len(sequence)))
        i_0 = 0
        
    mutant_indeces = random_mutation_generator(mutation_window, rate, num_mutants, number_fixed)
    if alph_type == "DNA":
        letters = np.array(["A", "C", "G", "T"])
    elif alph_type == "Numeric":
        letters = np.arange(4)
    else:
        raise ValueError("Alphabet type has to be either \"DNA\" or \"Numeric\"")
    
    if allowed_alph is not None:
        for i, x in enumerate(mutant_indeces):
            mutants[i + i_0] = sequence[0:site_start] + mutate_with_bias(mutation_window, x, letters, allowed_alph) + sequence[site_end:]
    else:
        for i, x in enumerate(mutant_indeces):
            mutants[i + i_0] = sequence[0:site_start] + mutate_from_index(mutation_window, x, letters) + sequence[site_end:]
        
    return mutants


@numba.njit
def _random_mutation_generator(sequence, rate, number_fixed):
    if number_fixed:
        num_mutations = int(rate*len(sequence))
    else:
        num_mutations = np.random.poisson(len(sequence) * rate)
    positions = np.random.choice(np.arange(len(sequence)), num_mutations, replace=False)
    mutants = np.random.choice(np.arange(3), num_mutations)
    return  [(x, y) for (x, y) in zip(positions, mutants)]
    

@numba.njit
def filter_letter(x, letter):
    return x != letter


@numba.njit
def filter_mutation(letter, alph):
    j = 0
    for i in range(alph.size):
        if filter_letter(alph[i], letter):
            j += 1
    result = np.empty(j, dtype=alph.dtype)
    j = 0
    for i in range(alph.size):
        if filter_letter(alph[i], letter):
            result[j] = alph[i]
            j += 1
    return result


@numba.njit
def choose_mutation_bias(letter, letters, allowed_alph):
    if letter == 'A':
        _alph = allowed_alph[0]
    elif letter == 'C':
        _alph = allowed_alph[1]
    elif letter == 'G':
        _alph = allowed_alph[2]
    elif letter == 'T':
        _alph = allowed_alph[3]
    
    alph = np.empty(np.sum(_alph), dtype=letters.dtype)
    i = 0
    j = 0
    for x in _alph:
        if x:
            alph[i] = letters[j]
            i += 1
        j += 1

    return alph

@numba.njit
def make_mutation(letter, letters, allowed_alph):
    alph = choose_mutation_bias(letter, letters, allowed_alph)
    index = np.random.choice(np.arange(len(alph)))
    return alph[index]