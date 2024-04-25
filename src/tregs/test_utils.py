import unittest
import numpy as np
from tregs.utils import smoothing

class test_utils(unittest.TestCase):
    
    def test_basic_smoothing(self):
        """Test smoothing with a standard odd window size."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([2, 3, 4, 5, 6, 7, 8])
        result = smoothing(data, windowsize=3)
        np.testing.assert_array_almost_equal(result, expected_output, decimal=1, err_msg="Basic smoothing failed")

    def test_even_window_size(self):
        """Test smoothing raises an error with an even window size."""
        data = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(RuntimeError):
            smoothing(data, windowsize=4)
    
    def test_minimum_array_size(self):
        """Test smoothing with the smallest array larger than the window size."""
        data = np.array([1, 2, 3, 4])
        result = smoothing(data, windowsize=3)
        expected_output = np.array([2, 3])  # Smooth over minimal slices
        np.testing.assert_array_almost_equal(result, expected_output, decimal=1, err_msg="Minimum array size smoothing failed")

    def test_non_uniform_data(self):
        """Test smoothing with non-uniform data distribution."""
        data = np.array([1, 1, 2, 6, 10, 10, 2, 1, 1])
        result = smoothing(data, windowsize=3)
        expected_output = np.array([1.33, 3.0, 6.0, 8.67, 7.33, 4.33, 1.33])  # Expected smoothed values
        np.testing.assert_array_almost_equal(result, expected_output, decimal=2, err_msg="Non-uniform data smoothing failed")

if __name__ == '__main__':
    unittest.main(verbosity=2)