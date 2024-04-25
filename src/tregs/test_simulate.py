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

    def test_sim_helper_empty_input(self):
        """ Test sim_helper with an empty list of mutants. """
        result = sim_helper([], lambda x: x, [])
        self.assertTrue(result.empty, "DataFrame should be empty for empty input mutants")

    def test_sim_helper_valid_input(self):
        """ Test sim_helper with a single mutant and a mock probability function. """
        mutants = ['ACGT']
        regions = [(0, 4)]
        result = sim_helper(mutants, lambda x: 0.5, regions)
        self.assertEqual(result.loc[0, 'pbound'], 0.5, "pbound should be 0.5 for the mock function")

    def test_sim_helper_assertion_error(self):
        """ Test sim_helper throws assertion error with out-of-bound regions. """
        mutants = ['ACGT']
        regions = [(0, 5)]  # Out-of-bound region
        with self.assertRaises(AssertionError):
            sim_helper(mutants, lambda x: [0.5], regions)

    def test_biased_mutants_typical_case(self):
        """ Test biased_mutants function with typical inputs. """
        seq = 'ACGTACGT'
        mutants = biased_mutants(seq, num_mutants=10, mutrate=0.1, allowed_alph=['A', 'C', 'G', 'T'])
        self.assertTrue(all(len(mut) == len(seq) for mut in mutants), "All mutants should have the same length as the input sequence")

    def test_biased_mutants_no_mutations(self):
        """ Test biased_mutants with zero mutation rate. """
        seq = 'ACGTACGT'
        mutants = biased_mutants(seq, num_mutants=10, mutrate=0, allowed_alph=['A', 'C', 'G', 'T'])
        self.assertEqual(len(set(mutants)), 1, "No mutations should result in one unique sequence")

    def test_biased_mutants_all_AT(self):
        """ Test biased_mutants with a sequence of only AT bases. """
        seq = 'ATATATAT'
        mutants = biased_mutants(seq, num_mutants=10, mutrate=0.1)
        self.assertTrue(all(mut == seq for mut in mutants), "All mutants should be the same as the original for all-AT input")

    def test_sim_preset_mutants(self):
        """ Test sim with preset mutants. """
        def mock_pbound(*args, **kwargs):
            return 0.1  # Return a constant probability for simplicity

        mutants = ['ACGT', 'TGCA']
        df = sim('ACGT', mock_pbound, ['A', 'C', 'G', 'T'], preset_mutants=mutants)
        self.assertEqual(len(df), len(mutants), "DataFrame should have as many rows as there are mutants")

    def test_dna_count_handling(self):
        """Test the handling of DNA counts and related DataFrame operations in the sim function."""
        promoter_seq = 'ACGTACGT'
        binding_site_seqs = ['ACGT']  # Example binding site sequences

        def mock_pbound(*args, **kwargs):
            return 0.5  # Return a constant probability for simplicity

        df = sim(promoter_seq, mock_pbound, binding_site_seqs, mutrate=0.5, num_mutants=1, scaling_factor=100)

        df['ct_1'] = [500, 750]  # Mock DNA counts

        # Assertions to verify correct DataFrame operations
        self.assertEqual(len(df), 2, "DataFrame should have entries for non-zero DNA counts")
        self.assertTrue((df['ct_1'] == [500, 750]).all(), "Check correct calculation of ct_1")
        self.assertTrue((df['norm_ct_1'] == [50.0, 50.0]).all(), "Check correct normalization of ct_1")


if __name__ == '__main__':
    unittest.main(verbosity=2)