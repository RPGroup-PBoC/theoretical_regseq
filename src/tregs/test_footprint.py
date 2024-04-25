import unittest
import numpy as np
from tregs.footprint import *

class TestGeneticAnalysisFunctions(unittest.TestCase):

    def test_match_seqs_no_mutations(self):
        """Sequences identical to the wild type should return all zeros."""
        wtseq = "AACCGGTT"
        mut_list = np.array(["AACCGGTT", "AACCGGTT"])
        expected = np.zeros((2, 8), dtype=bool)
        result = match_seqs(wtseq, mut_list)
        np.testing.assert_array_equal(result, expected)

    def test_match_seqs_multiple_mutations(self):
        """Correctly identifies positions of mutations."""
        wtseq = "AACCGGTT"
        mut_list = np.array(["TTCCGGTT", "AAACGGTA"])
        expected = np.array([
            [True, True, False, False, False, False, False, False],
            [False, False, True, False, False, False, False, True]
        ])
        result = match_seqs(wtseq, mut_list)
        np.testing.assert_array_equal(result, expected)

    def test_get_p_b_basic(self):
        """Calculate mutation probabilities correctly without pseudocounts."""
        all_mutarr = np.array([
            [False, True, False],
            [True, True, False]
        ])
        n_seqs = 2
        expected = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
        result = get_p_b(all_mutarr, n_seqs)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_p_b_with_pseudocounts(self):
        """Ensure pseudocounts are applied correctly."""
        all_mutarr = np.array([
            [False, False],
            [False, False],
            [False, False]
        ])
        n_seqs = 3
        pseudocount = 1
        expected = np.array([[0.75, 0.25], [0.75, 0.25]])
        result = get_p_b(all_mutarr, n_seqs, pseudocount=pseudocount)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_basic_binning(self):
        """Test basic functionality with straightforward data."""
        mu_data = pd.Series([1, 2, 5, 10, 15])
        nbins = 2
        upper_bound = np.median(mu_data)
        mu_bins, bin_cnt = bin_expression_levels(mu_data, nbins, upper_bound)
        self.assertEqual(len(mu_bins), 5)
        self.assertEqual(len(bin_cnt), 2)
        self.assertTrue((bin_cnt == [2, 3]).all())

    def test_bins_include_all_data(self):
        """Ensure all data points are correctly included in the bins."""
        mu_data = pd.Series([0, 5, 10, 15, 20])
        nbins = 4
        upper_bound = 20
        mu_bins, bin_cnt = bin_expression_levels(mu_data, nbins, upper_bound)
        self.assertEqual(bin_cnt.sum(), len(mu_data))

    def test_get_p_mu(self):
        """Test probability calculation from bin counts."""
        bin_cnt = np.array([10, 20, 30])
        n_seqs = 60
        expected = np.array([10/60, 20/60, 30/60])
        result = get_p_mu(bin_cnt, n_seqs)
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_joint_p(self):
        """Test the joint probability distribution calculation."""
        all_mutarr = np.array([[1, 0], [0, 1]])
        mu_bins = np.array([0, 1])
        nbins = 2
        len_promoter = 2
        result = get_joint_p(all_mutarr, mu_bins, nbins, len_promoter=len_promoter)
        expected = np.array([[[0, 1/2], [1/2, 0]], [[1/2, 0], [0, 1/2]]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mutual_information(self):
        """Tests basic MI computation with straightforward probabilities."""
        list_p_b = [np.array([0.5, 0.5]), np.array([0.7, 0.3])]
        p_mu = np.array([0.6, 0.4])
        list_joint_p = [np.array([[0.3, 0.2], [0.2, 0.3]]), np.array([[0.42, 0.28], [0.28, 0.02]])]
        len_promoter = 2
        result = MI(list_p_b, p_mu, list_joint_p, len_promoter)
        # Expected values need to be calculated based on the formula used in the function.
        # For simplicity, this is a placeholder for actual values you'd compute or verify against.
        expected_1 = 0.3 * np.log2(0.3 / (0.5 * 0.6)) + 0.2 * np.log2(0.2 / (0.5 * 0.4)) + 0.2 * np.log2(0.2 / (0.5 * 0.6)) + 0.3 * np.log2(0.3 / (0.5 * 0.4))
        expected_2 = 0.42 * np.log2(0.42 / (0.7 * 0.6)) + 0.28 * np.log2(0.28 / (0.7 * 0.4)) + 0.28 * np.log2(0.28 / (0.3 * 0.6)) + 0.02 * np.log2(0.02 / (0.3 * 0.4))
        expected = [expected_1, expected_2]
        np.testing.assert_almost_equal(result, expected, decimal=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)