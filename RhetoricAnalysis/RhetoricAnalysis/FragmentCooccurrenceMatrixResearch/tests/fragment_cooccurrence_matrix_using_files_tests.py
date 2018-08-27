import unittest

from RhetoricAnalysis.FragmentCooccurrenceMatrixResearch.code.fragment_cooccurrence_matrix import FragmentCooccurrenceMatrix


class FragmentCooccurrenceMatrixUsingFilesTests(unittest.TestCase):
    """
    Tests for `fragment_cooccurrence_matrix` when matrix is built from files, and also
    getting the fragments from unseen questions and answers.

    Attributes:
    - test_matrix: A test matrix instance from FragmentCooccurrenceMatrix.from_ccat_files using testing files
    """

    test_matrix = FragmentCooccurrenceMatrix.from_ccat_files(
        answer_arcs_filename='../resources/test_answer_arcs.json',
        question_arcs_filename='../resources/test_question_arcs.json'
    )

    def test_getting_number_of_question_fragments(self):
        """Test getting number of question fragments from matrix"""
        self.assertEqual(4, self.test_matrix.get_number_of_question_fragments())

    def test_getting_number_of_answer_fragments(self):
        """Test getting number of answer fragments from matrix"""
        self.assertEqual(2, self.test_matrix.get_number_of_answer_fragments())

    def test_getting_question_fragment_frequencies(self):
        """Test getting the occurrence frequencies for question fragments"""
        self.assertEqual(1, self.test_matrix.get_question_fragment_occurrence_frequency('does>*'))
        self.assertEqual(1, self.test_matrix.get_question_fragment_occurrence_frequency('accept_does'))
        self.assertEqual(1, self.test_matrix.get_question_fragment_occurrence_frequency('accept_*'))
        self.assertEqual(1, self.test_matrix.get_question_fragment_occurrence_frequency('accept_be'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_question_fragment_occurrence_frequency('q'))

    def test_getting_answer_fragment_frequencies(self):
        """Test getting the occurrence frequencies for answer fragments"""
        self.assertEqual(1, self.test_matrix.get_answer_fragment_occurrence_frequency('is_*'))
        self.assertEqual(1, self.test_matrix.get_answer_fragment_occurrence_frequency('is_more'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_answer_fragment_occurrence_frequency('f'))

    def test_getting_question_fragment_total_cooccurrence_frequencies(self):
        """Test getting total co occurrence frequencies for question fragments"""
        self.assertEqual(2, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('does>*'))
        self.assertEqual(2, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('accept_does'))
        self.assertEqual(2, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('accept_*'))
        self.assertEqual(2, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('accept_be'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_question_fragment_total_cooccurrence_frequency('not_here'))

    def test_getting_answer_fragment_total_cooccurrence_frequencies(self):
        """Test getting total co occurrence frequencies for answer fragments"""
        self.assertEqual(4, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('is_*'))
        self.assertEqual(4, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('is_more'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_answer_fragment_total_cooccurrence_frequency('not_here'))

    def test_getting_cooccurrence_frequency_for_question_answer_fragment_pair(self):
        """Test getting the co occurrence frequency for a given question-answer fragment pair"""
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('does>*', 'is_*'))
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('accept_does', 'is_*'))
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('accept_*', 'is_*'))
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('accept_be', 'is_*'))
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('does>*', 'is_more'))
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('accept_does', 'is_more'))
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('accept_*', 'is_more'))
        self.assertEqual(1, self.test_matrix.get_cooccurrence_frequency('accept_be', 'is_more'))
        # Test that a non-existent fragment gives None
        self.assertEqual(None, self.test_matrix.get_cooccurrence_frequency('not_here', 'is_more'))
        self.assertEqual(None, self.test_matrix.get_cooccurrence_frequency('not_here', 'is_more'))

    """
    At this point we are satisfied the cooccurrence matrix has been constructed correctly.
    We can infer from fragment_cooccurrence_matrix_tests that the methods will operate correctly. 
    """

    def test_getting_fragments_for_unseen_texts(self):
        """Test that we can get the correct fragments from unseen questions and answers"""
        unseen_question = "Plymouth has had some of the most significant new health investment under Labour-the first " \
                          "new dental hospital for 50 years and the first new medical hospital for 25 years. " \
                          "What plans does the Prime Minister have to protect the progress that has been " \
                          "made and the way in which waiting lists have plummeted?"
        unseen_question_fragments \
            = ['what>plans', 'have_plans', 'have_*', 'have_does', 'have_protect', 'what>*', 'have_what']

        unseen_answer = "By the time I met them they were all staunch Labour supporters, as a result of the message " \
                        "that we put to them. Yesterday I visited a number of places in Kent and asked people what " \
                        "the major issue affecting them was, and they said that they wanted to secure the recovery. " \
                        "I had to tell people that the Conservative party taking \u00a36 billion out of the economy " \
                        "would put the recovery at risk. The issue is very clear: jobs with Labour, unemployment " \
                        "under the Conservatives."
        unseen_answer_fragments = ['by>*', 'by_*', 'visited_*', 'said_*', 'asked>*', 'asked_was', 'asked_*',
                                   'said_wanted', 'had_tell', 'had_*', 'is_clear', 'is_*']

        returned_map = self.test_matrix.get_fragments_for_unseen_inputs(
            unseen_question=unseen_question,
            unseen_answer=unseen_answer,
            num_clusters=8,
            random_seed=125,
            ccat_verbose=False
        )

        self.assertEqual(len(unseen_question_fragments), len(returned_map['question_fragments']))
        self.assertEqual(len(unseen_answer_fragments), len(returned_map['answer_fragments']))

        for q_frag in returned_map['question_fragments']:
            self.assertNotEqual(-1, unseen_question_fragments.index(q_frag))

        for a_frag in returned_map['answer_fragments']:
            self.assertNotEqual(-1, unseen_answer_fragments.index(a_frag))


if __name__ == '__main__':
    unittest.main()
