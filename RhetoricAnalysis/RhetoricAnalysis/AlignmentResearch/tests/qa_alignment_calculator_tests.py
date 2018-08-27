import unittest
import nltk
from RhetoricAnalysis.AlignmentResearch.code import QAPairAlignmentCalculator


class FragmentCooccurrenceMatrixTests(unittest.TestCase):

    test_alignment_calculator = QAPairAlignmentCalculator(testing=True)

    def test_clean_data(self):
        """Test that cleaned text data is as expected"""
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        data = ["This is an outrage, the Government cannot possibly hope to maintain this position",
                "That a Minister of Parliament deems it necessary to lecture this collective of peers is laughable"]
        cleaned_data = self.test_alignment_calculator.clean_data(data, tokenizer)
        self.assertEqual(
            cleaned_data[0],
            ['is', 'an', 'outrage', 'the', 'government', 'cannot', 'possibly', 'hope', 'to', 'maintain', 'position']
        )
        self.assertEqual(
            cleaned_data[1],
            ['minister', 'of', 'parliament', 'deems', 'it', 'necessary', 'to', 'lecture', 'collective', 'of', 'peers',
             'is', 'laughable']
        )

    def test_calculate_euclidean_distance(self):
        """Test for correct calculation of euclidean distance and ValueError raised when necessary"""
        dist = self.test_alignment_calculator.calculate_euclidian_distance([10, 20], [5, 10])
        self.assertAlmostEqual(11.18034, dist, 5)
        dist = self.test_alignment_calculator.calculate_euclidian_distance([15, 1567], [12, 1800])
        self.assertAlmostEqual(233.01931, dist, 5)
        self.assertRaises(ValueError, self.test_alignment_calculator.calculate_euclidian_distance, [0, 0], [0])

    def test_calculate_alignment(self):
        """Test for correct calculation of alignment and ValueErrors raised when necessary"""
        alignment = self.test_alignment_calculator.calculate_alignment([10, 10, 10], [5, 5, 5], 5, 20)
        self.assertAlmostEqual(0.755983, alignment, 6)
        self.assertRaises(ValueError,
                          self.test_alignment_calculator.calculate_alignment,
                          [10, 10, 10], [5, 5, 5], 1000, 2000)
        self.assertRaises(ValueError,
                          self.test_alignment_calculator.calculate_alignment,
                          [10, 10, 10], [5, 5, 5], 0, 1)
        self.assertRaises(ValueError,
                          self.test_alignment_calculator.calculate_alignment,
                          [10, 10, 10], [5, 5, 5], 10, 1)
        self.assertRaises(ValueError,
                          self.test_alignment_calculator.calculate_alignment,
                          [10, 10, 10], [5, 5, 5], -10, 1)
        self.assertRaises(ValueError,
                          self.test_alignment_calculator.calculate_alignment,
                          [10, 10, 10], [5, 5, 5], 10, -1)

    def test_find_nearest_neighbours(self):
        """Test that finding the nearest neighbours works, using k = 1"""
        # First test in the question space
        # Test qid1's nearest neighbour is qid2
        nn_return_map = self.test_alignment_calculator.find_k_nearest_neighbours(
            self.test_alignment_calculator.question_vector_list[0],
            self.test_alignment_calculator.question_vector_list,
            1
        )
        self.assertEqual(nn_return_map['vectors'], [self.test_alignment_calculator.question_vector_list[1]])
        self.assertAlmostEqual(nn_return_map['distances'][0], 141.42136, 5)
        # Test qid2's nearest neighbour is qid1
        nn_return_map = self.test_alignment_calculator.find_k_nearest_neighbours(
            self.test_alignment_calculator.question_vector_list[1],
            self.test_alignment_calculator.question_vector_list,
            1
        )
        self.assertEqual(nn_return_map['vectors'], [self.test_alignment_calculator.question_vector_list[0]])
        self.assertAlmostEqual(nn_return_map['distances'][0], 141.42136, 5)

        # Now test in the answer space
        # Test aid1's nearest neighbour is aid3
        nn_return_map = self.test_alignment_calculator.find_k_nearest_neighbours(
            self.test_alignment_calculator.answer_vector_list[0],
            self.test_alignment_calculator.answer_vector_list,
            1
        )
        self.assertEqual(nn_return_map['vectors'], [self.test_alignment_calculator.answer_vector_list[2]])
        self.assertAlmostEqual(nn_return_map['distances'][0], 156.20499, 5)
        # Test aid3's nearest neighbour is aid1
        nn_return_map = self.test_alignment_calculator.find_k_nearest_neighbours(
            self.test_alignment_calculator.answer_vector_list[2],
            self.test_alignment_calculator.answer_vector_list,
            1
        )
        self.assertEqual(nn_return_map['vectors'], [self.test_alignment_calculator.answer_vector_list[0]])
        self.assertAlmostEqual(nn_return_map['distances'][0], 156.20499, 5)

    def test_calculate_qa_pair_alignment(self):
        """Test calculating the alignment of each qa-pair"""
        # In this case we get alignment of 0 because the nearest neighbour in the question space is the furthest in
        # the answer space and vice versa
        alignment = self.test_alignment_calculator.process_question_answer_pair_alignment(
            'qid1', 'aid1', 2, False, testing=True
        )
        self.assertAlmostEqual(alignment, 0.0, 1)
        # In this case we get alignment of 1 because the nearest neighbour in the question space is the nearest in
        # the answer space and vice versa
        alignment = self.test_alignment_calculator.process_question_answer_pair_alignment(
            'qid2', 'aid2', 2, False, testing=True
        )
        self.assertAlmostEqual(alignment, 1.0, 1)
        # In this case we get alignment of 0 because the nearest neighbour in the question space is the furthest in
        # the answer space and vice versa
        alignment = self.test_alignment_calculator.process_question_answer_pair_alignment(
            'qid3', 'aid3', 2, False, testing=True
        )
        self.assertAlmostEqual(alignment, 0.0, 1)
        # Test ValueError thrown when non-pair is passed
        self.assertRaises(
            ValueError,
            self.test_alignment_calculator.process_question_answer_pair_alignment,
            'qid3', 'aid2', 2, False, testing=True
        )


if __name__ == '__main__':
    unittest.main()
