from Bio import SeqIO
from tregs import seq_utils

import os
from distutils import dir_util
from pytest import fixture


@fixture
def datadir(tmpdir, request):

    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir

def test_find_binding_site(datadir):
    data = datadir.join('mg1655_genome.fasta')

    _genome = []
    for record in SeqIO.parse(data, "fasta"):
        _genome.append(str(record.seq))
    genome = _genome[0]

    lacO1 = seq_utils.get_regulatory_region(genome, 366343, reverse=True)
    lacWT = 'CAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGG'
    O1 = 'AATTGTGAGCGGATAACAATT'

    start, end = seq_utils.find_binding_site(lacO1, lacWT)
    assert lacO1[start:end] == lacWT

    start, end = seq_utils.find_binding_site(lacO1, O1)
    assert lacO1[start:end] == O1