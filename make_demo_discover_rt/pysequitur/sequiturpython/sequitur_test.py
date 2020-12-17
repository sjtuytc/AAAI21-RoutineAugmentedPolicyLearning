import unittest
from grammar import Grammar

class TestSequitur(unittest.TestCase):
    def test_sequitur(self):
        """docstring for test_sequitur"""
        g = Grammar()
        g.train_string("Hello, world!")
        
        self.assertEqual("0 --(0)--> H e l l o , _ w o r l d ! \n", g.print_grammar())

    def test_sequitur_base(self):
        """docstring for test_sequitur_base"""
        g = Grammar()
        g.train_string("abcabdabcabd")

        self.assertEqual("0 --(0)--> 1 1 \n1 --(2)--> 2 c 2 d                                       abcabd\n2 --(2)--> a b                                           ab\n", g.print_grammar())

if __name__ == '__main__':
    unittest.main()
