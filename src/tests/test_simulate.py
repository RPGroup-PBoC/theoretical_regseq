import numpy as np
from tregseq import simulate

def test_get_d_energy1():
    testmat = np.stack([np.arange(5)] * 4)
    assert simulate.get_d_energy('A' * 5, testmat) == 10
    assert simulate.get_d_energy('T' * 5, testmat) == 10
    assert simulate.get_d_energy('ACGCT', testmat) == 10


def test_get_d_energy2():
    testmat = np.stack([[0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2],
                        [3, 3, 3, 3, 3]])
    assert simulate.get_d_energy('A' * 5, testmat) == 0
    assert simulate.get_d_energy('T' * 5, testmat) == 15
    assert simulate.get_d_energy('ACGCT', testmat) == 7
