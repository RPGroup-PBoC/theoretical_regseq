from Bio.Seq import Seq

def get_regulatory_region(genome, TSS, reverse=False):
    '''
    given the full genomic sequence and the TSS, find the regulatory region,
    defined as the 115 bases upstream of the transcription start site
    and 45 bases downstream

    Args:
        genome (str): full genomic sequence
        TSS (int): transcription start site
        reverse (bool, optional): whether the regulatory region is on the
        reverse strand of the genomic DNA. defaults to False.

    Returns:
        str: 160bp sequence of the regulatory region
    '''    

    if not reverse:
        region = genome[(TSS-115):(TSS+45)]
    else:
        _region = genome[(TSS-45):(TSS+115)]
        region = str(Seq(_region).reverse_complement())

    return region


def seq_to_list(seq):
    '''
    map each nucleotide in a DNA sequence to an integer from 0 to 3 and outputs
    the list of integers

    Args:
        seq (str): DNA sequence

    Returns:
        list: list of integers corresponding the the DNA sequence
    '''

    mat = []
    seq_dict = {'A':0, 'C':1, 'G':2, 'T':3}
    for base in seq:
        mat.append(seq_dict[base])

    return mat


def find_binding_site(region, binding_site):
    '''
    given the sequence of the promoter region and the sequence of the binding
    sites, find the start and end indices of the binding site

    Args:
        region (str): sequence of the promoter region
        binding_site (str): sequence of the binding site

    Returns:
        int, int: start index, end index
    '''    

    start = region.find(binding_site)
    end = start + len(binding_site)

    return start, end
