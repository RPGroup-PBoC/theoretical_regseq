import unittest
import numpy as np
from tregs.simulate import *

class test_simulate(unittest.TestCase):

    def test_fix_wt_identity(self):
        matrix = np.array([
            [0.0, 0.2, 0.3, 0.4],
            [0.1, 0.0, 0.3, 0.4],
            [0.1, 0.2, 0.0, 0.4],
            [0.1, 0.2, 0.3, 0.0]
        ])
        seq = "ACGT"
        expected = np.array([
            [0.0, 0.2, 0.3, 0.4],
            [0.1, 0.0, 0.3, 0.4],
            [0.1, 0.2, 0.0, 0.4],
            [0.1, 0.2, 0.3, 0.0]
        ])
        result = fix_wt(matrix, seq)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fix_wt_non_identity(self):
        matrix = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.4, 0.3],
            [0.3, 0.4, 0.1, 0.2],
            [0.4, 0.3, 0.2, 0.1]
        ])
        seq = "TGCA"
        expected = np.array([
            [-0.3, -0.2, -0.1, 0],
            [-0.2, -0.3, 0, -0.1],
            [-0.1, 0, -0.3, -0.2],
            [0, -0.1, -0.2, -0.3]
        ])
        result = fix_wt(matrix, seq)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_d_energy(self):
        energy_mat = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ])
        seq = "ACGT"
        self.assertEqual(get_d_energy(seq, energy_mat), 6)

    def test_get_d_energy_with_e_wt(self):
        energy_mat = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ])
        seq = "ACGT"
        self.assertEqual(get_d_energy(seq, energy_mat, e_wt=-1), 5)

    def test_get_weight(self):
        energy_mat = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ])
        seq = "ACGT"
        expected = np.exp(-6)
        self.assertAlmostEqual(get_weight(seq, energy_mat), expected)

    def test_get_weight_with_wt(self):
        energy_mat = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ])
        seq = "ACGT"
        expected = np.exp(-7)
        self.assertAlmostEqual(get_weight(seq, energy_mat, e_wt=1), expected)

    def test_generate_emap_fixed(self):
        seq = "ACGT"
        emat = generate_emap(seq, fixed=True, fixed_value=0.5)
        expected_emat = np.array([
            [0, 0.5, 0.5, 0.5],
            [0.5, 0, 0.5, 0.5],
            [0.5, 0.5, 0, 0.5],
            [0.5, 0.5, 0.5, 0]
        ])
        self.assertTrue((emat == expected_emat).all())

    def test_generate_emap_random(self):
        seq = "AAAA"
        emat = generate_emap(seq, fixed=False, max_mut_energy=0.5)
        self.assertTrue((emat[1, :] >= 0.1).all() and (emat[1, :] <= 0.5).all())

    def test_get_dna_cnt(self):
        n_seqs = 10
        results = get_dna_cnt(n_seqs)
        self.assertEqual(len(results), n_seqs)
        self.assertTrue((results >= 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)