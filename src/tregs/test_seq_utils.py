import unittest
from Bio.Seq import Seq

from tregs.seq_utils import get_regulatory_region, seq_to_list, find_binding_site

class test_seq_utils(unittest.TestCase):

    # Tests for get_regulatory_region
    def test_get_regulatory_region_forward(self):
        genome = "A" * 100 + "C" * 160 + "G" * 100  # 360 bp genome
        TSS = 150
        result = get_regulatory_region(genome, TSS)
        self.assertEqual(result, "A" * 65 + "C" * 95, "Forward strand regulatory region incorrect")

    def test_get_regulatory_region_reverse(self):
        genome = "A" * 100 + "C" * 160 + "G" * 100
        TSS = 150
        result = get_regulatory_region(genome, TSS, reverse=True)
        expected = str(Seq("C" * 155 + "G" * 5).reverse_complement())
        self.assertEqual(result, expected, "Reverse strand regulatory region incorrect")

    def test_get_regulatory_region_boundary_conditions(self):
        genome = "A" * 160
        TSS = 115  # Near the start of the genome
        result = get_regulatory_region(genome, TSS)
        self.assertEqual(result, "A" * 160, "Boundary condition handling failed")

    # Tests for seq_to_list
    def test_seq_to_list_simple(self):
        seq = "ACGT"
        result = seq_to_list(seq)
        self.assertEqual(result, [0, 1, 2, 3], "Simple sequence conversion failed")

    def test_seq_to_list_repeated_chars(self):
        seq = "AAAACCCC"
        result = seq_to_list(seq)
        self.assertEqual(result, [0, 0, 0, 0, 1, 1, 1, 1], "Repeated characters conversion failed")

    def test_seq_to_list_empty_string(self):
        seq = ""
        result = seq_to_list(seq)
        self.assertEqual(result, [], "Empty string should return empty list")

    # Tests for find_binding_site
    def test_find_binding_site_present(self):
        region = "ATCGATCG"
        binding_site = "CGA"
        start, end = find_binding_site(region, binding_site)
        self.assertEqual((start, end), (2, 5), "Finding present binding site failed")

    def test_find_binding_site_absent(self):
        region = "ATCGATCG"
        binding_site = "GTT"
        start, end = find_binding_site(region, binding_site)
        self.assertEqual((start, end), (-1, -1), "Handling absent binding site failed")

    def test_find_binding_site_multiple_occurrences(self):
        region = "ATCGATCGATCG"
        binding_site = "ATC"
        start, end = find_binding_site(region, binding_site)
        self.assertEqual((start, end), (0, 3), "Finding first occurrence of binding site failed")

if __name__ == '__main__':
    unittest.main(verbosity=2)