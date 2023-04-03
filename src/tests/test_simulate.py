import unittest
import os
import numpy as np
from Bio import SeqIO

import tregseq
from tregseq import simulate

#dir_path = os.path.dirname(os.path.realpath(__file__))
#conf_path = os.path.join(dir_path, 'mg1655_genome.fasta')


class TestModel(unittest.TestCase):
    
    def test_get_d_energy1(self):
        testmat = np.stack([np.arange(5)] * 4)
        self.assertTrue(simulate.get_d_energy('A' * 5, testmat) == 10)
        self.assertTrue(simulate.get_d_energy('T' * 5, testmat) == 10)
        self.assertTrue(simulate.get_d_energy('ACGCT', testmat) == 10)
    
    
    def test_get_d_energy2(self):
        testmat = np.stack([[0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1],
                            [2, 2, 2, 2, 2],
                            [3, 3, 3, 3, 3]])
        self.assertTrue(simulate.get_d_energy('A' * 5, testmat) == 0)
        self.assertTrue(simulate.get_d_energy('T' * 5, testmat) == 15)
        self.assertTrue(simulate.get_d_energy('ACGCT', testmat) == 7)
    
    '''
    def test_find_binding_site(self):
        _genome = []
        for record in SeqIO.parse(conf_path, "fasta"):
            _genome.append(str(record.seq))
        genome = _genome[0]

        lacO1 = simulate.get_regulatory_region(genome, 366343, reverse=True)
        lacWT = 'CAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGG'
        O1 = 'AATTGTGAGCGGATAACAATT'

        start, end = simulate.find_binding_site(lacO1, lacWT)
        self.assertTrue(lacO1[start:end] == lacWT)

        start, end = simulate.find_binding_site(lacO1, O1)
        self.assertTrue(lacO1[start:end] == O1)
    '''

if __name__ == '__main__':
    unittest.main(verbosity=3)
