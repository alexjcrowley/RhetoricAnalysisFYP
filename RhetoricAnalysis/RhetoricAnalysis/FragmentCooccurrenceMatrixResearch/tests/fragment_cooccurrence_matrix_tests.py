import unittest
from RhetoricAnalysis.FragmentCooccurrenceMatrixResearch.code.fragment_cooccurrence_matrix import FragmentCooccurrenceMatrix


class FragmentCooccurrenceMatrixTests(unittest.TestCase):
    """
    Tests for `fragment_cooccurrence_matrix`.

    Attributes:
    - test_matrix: A test matrix instance from FragmentCooccurrenceMatrix.test_matrix
    """

    test_matrix = FragmentCooccurrenceMatrix.test_matrix()

    def test_getting_number_of_question_fragments(self):
        """Test getting number of question fragments from matrix"""
        self.assertEqual(5, self.test_matrix.get_number_of_question_fragments())

    def test_getting_number_of_answer_fragments(self):
        """Test getting number of answer fragments from matrix"""
        self.assertEqual(5, self.test_matrix.get_number_of_answer_fragments())

    def test_getting_question_fragment_frequencies(self):
        """Test getting the occurrence frequencies for question fragments"""
        self.assertEqual(40, self.test_matrix.get_question_fragment_occurrence_frequency('v'))
        self.assertEqual(100, self.test_matrix.get_question_fragment_occurrence_frequency('w'))
        self.assertEqual(20, self.test_matrix.get_question_fragment_occurrence_frequency('x'))
        self.assertEqual(10, self.test_matrix.get_question_fragment_occurrence_frequency('y'))
        self.assertEqual(5, self.test_matrix.get_question_fragment_occurrence_frequency('z'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_question_fragment_occurrence_frequency('q'))

    def test_getting_answer_fragment_frequencies(self):
        """Test getting the occurrence frequencies for answer fragments"""
        self.assertEqual(25, self.test_matrix.get_answer_fragment_occurrence_frequency('a'))
        self.assertEqual(50, self.test_matrix.get_answer_fragment_occurrence_frequency('b'))
        self.assertEqual(15, self.test_matrix.get_answer_fragment_occurrence_frequency('c'))
        self.assertEqual(7, self.test_matrix.get_answer_fragment_occurrence_frequency('d'))
        self.assertEqual(8, self.test_matrix.get_answer_fragment_occurrence_frequency('e'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_answer_fragment_occurrence_frequency('f'))

    def test_getting_question_fragment_total_cooccurrence_frequencies(self):
        """Test getting total co occurrence frequencies for question fragments"""
        self.assertEqual(67, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('v'))
        self.assertEqual(37, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('w'))
        self.assertEqual(14, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('x'))
        self.assertEqual(81, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('y'))
        self.assertEqual(15, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('z'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('q'))

    def test_getting_answer_fragment_total_cooccurrence_frequencies(self):
        """Test getting total co occurrence frequencies for answer fragments"""
        self.assertEqual(39, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('a'))
        self.assertEqual(42, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('b'))
        self.assertEqual(37, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('c'))
        self.assertEqual(38, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('d'))
        self.assertEqual(58, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('e'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('f'))

    def test_getting_question_fragment_ids(self):
        """Test getting the ids for question fragments"""
        self.assertEqual(0, self.test_matrix.get_question_fragment_id('v'))
        self.assertEqual(1, self.test_matrix.get_question_fragment_id('w'))
        self.assertEqual(2, self.test_matrix.get_question_fragment_id('x'))
        self.assertEqual(3, self.test_matrix.get_question_fragment_id('y'))
        self.assertEqual(4, self.test_matrix.get_question_fragment_id('z'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_question_fragment_id('q'))

    def test_getting_answer_fragment_ids(self):
        """Test getting the ids for answer fragments"""
        self.assertEqual(0, self.test_matrix.get_answer_fragment_id('a'))
        self.assertEqual(1, self.test_matrix.get_answer_fragment_id('b'))
        self.assertEqual(2, self.test_matrix.get_answer_fragment_id('c'))
        self.assertEqual(3, self.test_matrix.get_answer_fragment_id('d'))
        self.assertEqual(4, self.test_matrix.get_answer_fragment_id('e'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_answer_fragment_id('f'))

    def test_getting_answer_fragment_columns(self):
        """Test getting the answer fragment for the given column"""
        self.assertEqual('a', self.test_matrix.get_answer_fragment_for_column(0))
        self.assertEqual('b', self.test_matrix.get_answer_fragment_for_column(1))
        self.assertEqual('c', self.test_matrix.get_answer_fragment_for_column(2))
        self.assertEqual('d', self.test_matrix.get_answer_fragment_for_column(3))
        self.assertEqual('e', self.test_matrix.get_answer_fragment_for_column(4))
        # Test that a non-existent column gives None
        self.assertEqual(None, self.test_matrix.get_answer_fragment_for_column(5))

    def test_getting_question_fragment_rows(self):
        """Test getting the question fragment for the given row"""
        self.assertEqual('v', self.test_matrix.get_question_fragment_for_row(0))
        self.assertEqual('w', self.test_matrix.get_question_fragment_for_row(1))
        self.assertEqual('x', self.test_matrix.get_question_fragment_for_row(2))
        self.assertEqual('y', self.test_matrix.get_question_fragment_for_row(3))
        self.assertEqual('z', self.test_matrix.get_question_fragment_for_row(4))
        # Test that a non-existent row gives None
        self.assertEqual(None, self.test_matrix.get_question_fragment_for_row(5))

    def test_getting_question_fragment_coccurrence_frequencies(self):
        """Test getting all co occurrence frequencies for the given question fragment"""
        self.assertEqual([12, 11, 10, 14, 20], self.test_matrix.get_all_coocurrence_frequencies_for_question_fragment('v'))
        self.assertEqual([7, 11, 7, 8, 4], self.test_matrix.get_all_coocurrence_frequencies_for_question_fragment('w'))
        self.assertEqual([3, 1, 0, 0, 10], self.test_matrix.get_all_coocurrence_frequencies_for_question_fragment('x'))
        self.assertEqual([17, 16, 15, 16, 17], self.test_matrix.get_all_coocurrence_frequencies_for_question_fragment('y'))
        self.assertEqual([0, 3, 5, 0, 7], self.test_matrix.get_all_coocurrence_frequencies_for_question_fragment('z'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_all_coocurrence_frequencies_for_question_fragment('q'))

    def test_getting_answer_fragment_coccurrence_frequencies(self):
        """Test getting all co occurrence frequencies for the given answer fragment"""
        self.assertEqual([12, 7, 3, 17, 0], self.test_matrix.get_all_coocurrence_frequencies_for_answer_fragment('a'))
        self.assertEqual([11, 11, 1, 16, 3], self.test_matrix.get_all_coocurrence_frequencies_for_answer_fragment('b'))
        self.assertEqual([10, 7, 0, 15, 5], self.test_matrix.get_all_coocurrence_frequencies_for_answer_fragment('c'))
        self.assertEqual([14, 8, 0, 16, 0], self.test_matrix.get_all_coocurrence_frequencies_for_answer_fragment('d'))
        self.assertEqual([20, 4, 10, 17, 7], self.test_matrix.get_all_coocurrence_frequencies_for_answer_fragment('e'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_all_coocurrence_frequencies_for_answer_fragment('f'))

    def test_getting_cooccurrence_frequency_for_question_answer_fragment_pair(self):
        """Test getting the co occurrence frequency for a given question-answer fragment pair"""
        self.assertEqual(12, self.test_matrix.get_cooccurrence_frequency('v', 'a'))
        self.assertEqual(11, self.test_matrix.get_cooccurrence_frequency('w', 'b'))
        self.assertEqual(0, self.test_matrix.get_cooccurrence_frequency('x', 'c'))
        self.assertEqual(16, self.test_matrix.get_cooccurrence_frequency('y', 'd'))
        self.assertEqual(7, self.test_matrix.get_cooccurrence_frequency('z', 'e'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_cooccurrence_frequency('q', 'e'))
        self.assertEqual(None, self.test_matrix.get_cooccurrence_frequency('z', 'f'))

    def test_getting_summed_cooccurrence_frequencies_for_question_fragment_lists(self):
        """Test getting the summed co occurrence frequencies for question fragment lists"""
        self.assertEqual(
            [19, 22, 17, 22, 24],
            self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['v', 'w'])['in_matrix'])
        self.assertEqual(
            [19, 22, 17, 22, 24],
            self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['w', 'v'])['in_matrix'])
        self.assertEqual(
            [20, 20, 20, 16, 34],
            self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['x', 'y', 'z'])['in_matrix'])
        self.assertEqual(
            [19, 22, 17, 22, 24],
            self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['v', 'w', 'q'])['in_matrix'])
        self.assertEqual(
            ['q'],
            self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['v', 'w', 'q'])['not_in_matrix'])

    def test_getting_summed_cooccurrence_frequencies_for_answer_fragment_lists(self):
        """Test getting the summed co occurrence frequencies for answer fragment lists"""
        self.assertEqual([23, 18, 4, 33, 3],
                         self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['a', 'b'],
                         question_fragments=False)['in_matrix'])
        self.assertEqual([23, 18, 4, 33, 3],
                         self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['b', 'a'],
                         question_fragments=False)['in_matrix'])
        self.assertEqual([44, 19, 10, 48, 12],
                         self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['c', 'd', 'e'],
                         question_fragments=False)['in_matrix'])
        self.assertEqual([23, 18, 4, 33, 3],
                         self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['a', 'b', 'f'],
                         question_fragments=False)['in_matrix'])
        self.assertEqual(['f'],
                         self.test_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(['a', 'b', 'f'],
                         question_fragments=False)['not_in_matrix'])

    def test_getting_summed_relative_cooccurrence_frequencies_for_question_fragment_lists(self):
        """Test getting the summed relative co occurrence frequencies for question fragment lists"""
        expected_values = [8.8785, 10.2804, 7.9439, 10.2803, 11.2150]
        test_values = self.test_matrix.get_summed_relative_cooccurrence_frequencies_for_fragment_list(
            ['v', 'w', 'q'])

        self.assertEqual(test_values['not_in_matrix'], ['q'])

        for i in range(0, len(expected_values)):
            self.assertAlmostEqual(expected_values[i], test_values['in_matrix'][i], 3)

        expected_values = [9.3458, 9.3458, 9.3458, 7.4766, 15.8879]
        test_values = self.test_matrix.get_summed_relative_cooccurrence_frequencies_for_fragment_list(
            ['x', 'y', 'z', 'q'])

        self.assertEqual(test_values['not_in_matrix'], ['q'])

        for i in range(0, len(expected_values)):
            self.assertAlmostEqual(expected_values[i], test_values['in_matrix'][i], 3)

    def test_getting_summed_relative_cooccurrence_frequencies_for_answer_fragment_lists(self):
        """Test getting the summed relative co occurrence frequencies for answer fragment lists"""
        expected_values = [10.7477, 8.4112, 1.8692, 15.4206, 1.4019]
        test_values = self.test_matrix.get_summed_relative_cooccurrence_frequencies_for_fragment_list(
            ['a', 'b', 'f'], question_fragments=False)

        self.assertEqual(test_values['not_in_matrix'], ['f'])

        for i in range(0, len(expected_values)):
            self.assertAlmostEqual(expected_values[i], test_values['in_matrix'][i], 3)

        expected_values = [20.5607, 8.8785, 4.6729, 22.4299, 5.6075]
        test_values = self.test_matrix.get_summed_relative_cooccurrence_frequencies_for_fragment_list(
            ['c', 'd', 'e', 'f'], question_fragments=False)

        self.assertEqual(test_values['not_in_matrix'], ['f'])

        for i in range(0, len(expected_values)):
            self.assertAlmostEqual(expected_values[i], test_values['in_matrix'][i], 3)

    def test_get_defined_number_of_greatest_fragments_for_question_fragment_list(self):
        """Test getting the most frequent answer fragments for a list of question fragments"""
        self.assertEqual({'e': 24, 'b': 22, 'd': 22},
                         self.test_matrix.get_defined_number_of_greatest_fragments_for_fragment_list(['v', 'w'],
                         limit=3)['in_matrix'])
        self.assertEqual({'e': 34, 'a': 20, 'b': 20},
                         self.test_matrix.get_defined_number_of_greatest_fragments_for_fragment_list(['x', 'y', 'z'],
                         limit=3)['in_matrix'])
        self.assertEqual(['q'],
                         self.test_matrix.get_defined_number_of_greatest_fragments_for_fragment_list(['v', 'w', 'q'],
                         limit=3)['not_in_matrix'])

    def test_get_defined_number_of_greatest_fragments_for_answer_fragment_list(self):
        """Test getting the most frequent question fragments for a list of answer fragments"""
        self.assertEqual({'y': 33, 'v': 23, 'w': 18},
                         self.test_matrix.get_defined_number_of_greatest_fragments_for_fragment_list(['a', 'b'],
                         limit=3,
                         question_fragments=False)['in_matrix'])
        self.assertEqual({'y': 48, 'v': 44, 'w': 19},
                         self.test_matrix.get_defined_number_of_greatest_fragments_for_fragment_list(['c', 'd', 'e'],
                         limit=3,
                         question_fragments=False)['in_matrix'])
        self.assertEqual(['f'],
                         self.test_matrix.get_defined_number_of_greatest_fragments_for_fragment_list(['c', 'd', 'e', 'f'],
                         limit=3,
                         question_fragments=False)['not_in_matrix'])

    def test_is_single_world_fragment(self):
        """Test determining if a fragment contains only a single word"""
        self.assertEqual(True, self.test_matrix.is_single_word_fragment("is>*"))
        self.assertEqual(True, self.test_matrix.is_single_word_fragment("is_*"))
        self.assertEqual(False, self.test_matrix.is_single_word_fragment("is_aware"))
        self.assertEqual(False, self.test_matrix.is_single_word_fragment("is>aware"))

    def test_is_composite_stop_word_fragment(self):
        """Test determining if a fragment is made up only of stop words"""
        self.assertEqual(True, self.test_matrix.is_composite_stop_word_fragment(['is', 'there', 'aware', 'is_there'], 'is_there'))
        self.assertEqual(False, self.test_matrix.is_composite_stop_word_fragment(['is', 'there', 'aware', 'is_there'], 'am_aware'))


if __name__ == '__main__':
    unittest.main()
