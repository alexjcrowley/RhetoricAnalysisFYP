import gensim
import json

import itertools
import nltk
import math
import sys
import time
import statistics
from multiprocessing import Pool as ThreadPool
from collections import OrderedDict
from types import SimpleNamespace


class LabeledLineSentence:
    """
    Helper class for usage with the Gensim Doc2Vec model to allow easy vocab passing by generating a labelled sentence
    for each of the

    Attribute:
    - labels_list: The list of labels/ids for the documents where the label at index x is for the document at index x
    - doc_list: The list of documents to be matched with labels
    """

    labels_list = None
    doc_list = None

    def __init__(self, doc_list, labels_list):
        """
        Initialise the object with the relevant doc_list and labels_list.

        :param doc_list: The list of documents to be matched to their labels
        :param labels_list: The list of labels to be matched to their documents
        :raises ValueError: If the label_list and doc_list are of different lengths
        """
        if len(doc_list) != len(labels_list):
            raise ValueError("Document list and labels list must be of the same length")
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        """
        Iterate over all the documents and create a LabeledSentence for the document using it's id from the label list
        """
        for index in range(0, len(self.doc_list)):
            yield gensim.models.doc2vec.LabeledSentence(self.doc_list[index], [self.labels_list[index]])


class QAPairAlignmentCalculator:
    """
    Class to allow for the creation of vector spaces given a json corpus to calculate the alignment of question-answer
    pairs.

    Attributes:
    - model: The Gensim Doc2Vec model to be used
    - alignments: The array of all alignment values calculated for the passed corpus
    - question_id_list: The array of all the question ids from the corpus
    - question_vector_list: The array of vectors for each question from the corpus
    - answer_id_list: The array of all the answer ids from the corpus
    - answer_vector_list: The array of vectors for each question from the corpus
    - question_to_answer_map: Dictionary containing question ids mapped to the answer id of the answer for the question
    - question_to_vector_map: Dictionary mapping the question ids to the vectors for the question
    - answer_to_vector_map: Dictionary mapping the answer ids to the vectors for the answers
    - stop_words: Set of stopwords to be removed from questions and answers
    """

    model = None
    alignments = None
    question_id_list = None
    question_vector_list = None
    answer_id_list = None
    answer_vector_list = None
    question_to_answer_map = None
    question_to_vector_map = None
    answer_to_vector_map = None
    stop_words = None

    def __init__(self, corpus_filepath=None, model_filepath=None, vector_size=100, min_count=200, epochs=100, workers=8,
                 stop_words_filepath=None, verbose=False, testing=False):
        """
        Initialise the object by either training a model using the corpus at the given file path and the param values
        given or using an existing model from model_filepath.

        Corpus requirements:
        - Must be an array of JSON objects representing questions and answers.
        - Questions and answers must have a top level 'text' tag
        - Questions must have a top level 'id' tag.
        - Answers must have a top level 'reply-to' tag allowing an answer to be linked to the invoking question.

        :param corpus_filepath: The file path for the corpus to be used
        :param model_filepath: The (optional) file path for the previously saved model, without this a new model is
                               trained
        :param vector_size: The size of the Doc2Vec vectors to use (This is a gensim parameter)
        :param min_count: The minimum number of times a word must occur to be considered (This is a gensim parameter)
        :param epochs: The number of epochs for which to train the model (This is a gensim parameter)
        :param workers: The number of workers to use during model training, align to the number of cores on the computer
                        for optimal training times (This is a gensim parameter)
        :param stop_words_filepath: The file path to use to build the stop words set
        :param verbose: Whether or not to print details about the process
        :param testing: Whether or not to just return the basic test object and vector space
        :raises ValueError: If duplicate vectors are found or no corpus filepath was given
        """

        if testing:
            self.question_id_list = ['qid1', 'qid2', 'qid3']
            self.question_vector_list = [[100, 200, 100], [200, 300, 100], [120, 300, 225]]
            self.answer_id_list = ['aid1', 'aid2', 'aid3']
            self.answer_vector_list = [[150, 250, 50], [50, 350, 170], [30, 150, 50]]
            self.question_to_answer_map = {'qid1': 'aid1', 'qid2': 'aid2', 'qid3': 'aid3'}
            self.question_to_vector_map = {self.question_id_list[0]: self.question_vector_list[0],
                                           self.question_id_list[1]: self.question_vector_list[1],
                                           self.question_id_list[2]: self.question_vector_list[2]}
            self.answer_to_vector_map = {self.answer_id_list[0]: self.answer_vector_list[0],
                                         self.answer_id_list[1]: self.answer_vector_list[1],
                                         self.answer_id_list[2]: self.answer_vector_list[2]}
            self.model = SimpleNamespace()
            self.model.docvecs = {**self.question_to_vector_map, **self.answer_to_vector_map}
            self.stop_words = {'this', 'a', 'that'}
            return

        if corpus_filepath is None:
            raise ValueError("Parameter corpus_filepath must be given")

        if verbose:
            print("Using corpus filepath: {}".format(corpus_filepath))
            print("Using model filepath: {}".format(model_filepath))

        start = time.time()

        # Using ordered dicts allows us to have a deterministic sequence of output later on when processing alignments
        self.question_to_answer_map = OrderedDict()
        self.question_to_vector_map = OrderedDict()
        self.answer_to_vector_map = OrderedDict()
        self.stop_words = set()

        # If stop words have been given then initialise the set of stop words using the .txt file
        if stop_words_filepath is not None:
            with open(stop_words_filepath) as stop_words_lines:
                for line in stop_words_lines:
                    line = line.replace("\n", "")
                    self.stop_words.add(line)
                stop_words_lines.close()

        with open(corpus_filepath) as corpus_file:
            corpus_object = json.load(corpus_file)
            corpus_file.close()

        doc_labels = []
        text = []

        for qa_object in corpus_object:
            if qa_object['is_answer']:
                self.question_to_answer_map[qa_object['reply-to']] = qa_object['id']
            doc_labels.append(qa_object['id'])
            text.append(qa_object['text'])

        if verbose:
            print("Total number of QA pairs {}".format(len(self.question_to_answer_map)))

        if model_filepath is None:
            tokenizer = nltk.RegexpTokenizer(r'\w+')
            # Model filepath not given so we need to create the model
            if verbose:
                print("Finished loading questions and answers from corpus, cleaning text")
            text = self.clean_data(text, tokenizer)
            if verbose:
                print("Finished cleaning text, training model")
            model_iterator = LabeledLineSentence(text, doc_labels)
            self.model = gensim.models.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs,
                                               workers=workers)
            self.model.build_vocab(model_iterator)
            self.model.train(model_iterator, total_examples=self.model.corpus_count, epochs=self.model.iter)
        else:
            # Model filepath given so load the model
            if verbose:
                print("Loading model from filepath: {}".format(model_filepath))
            self.model = gensim.models.doc2vec.Doc2Vec.load(model_filepath)
            if verbose:
                print("Model loading complete")

        if verbose:
            print("Initialising data for alignment calculation readiness")

        # Model has been trained or loaded from a previously saved model, now move on to filling the required
        # data structures
        self.question_id_list = []
        self.question_vector_list = []
        self.answer_id_list = []
        self.answer_vector_list = []
        self.alignments = []

        all_vectors = []

        for q_id in self.question_to_answer_map.keys():
            a_id = self.question_to_answer_map[q_id]
            q_vec = self.model.docvecs[q_id].tolist()
            a_vec = self.model.docvecs[a_id].tolist()

            try:
                # If we can find the index of this vector that means we have seen it before which will break the model
                all_vectors.index(q_vec)
                raise ValueError("Duplicate vector {} found for question id: {}".format(q_vec, q_id))
            except ValueError:
                all_vectors.append(q_vec)
                pass

            try:
                # If we can find the index of this vector that means we have seen it before which will break the model
                all_vectors.index(a_vec)
                raise ValueError("Duplicate vector {} found for answer id: {}".format(a_vec, a_id))
            except ValueError:
                all_vectors.append(a_vec)
                pass

            self.question_id_list.append(q_id)
            self.question_vector_list.append(q_vec)
            self.answer_id_list.append(a_id)
            self.answer_vector_list.append(a_vec)

        end = time.time()
        if verbose:
            print("Initialisation complete in {} seconds".format(end - start))

    def save_all_pair_alignments_to_file(self, file_path):
        """
        Save the pair alignments as a CSV where each row contains: pair_id, pair_alignment. File is saved to
        pair_alignments.csv

        :param file_path: The file path to save the model to
        """
        with open(file_path, "w") as file:
            for idx in range(0, len(self.question_id_list)):
                pair_id = "{}-q-a-{}".format(self.question_id_list[idx], self.answer_id_list[idx])
                file.write("{}, {}\n".format(pair_id, self.alignments[idx]))
            file.close()

    def process_all_pair_alignments(self, k=5, threads=4, verbose=False):
        """
        Iterate over all the question-answer pairs and calculate their alignment

        :param k: The number of nearest neighbours to consider in the alignment calculations
        :param threads: The number of threads to use in the alignment
        :param verbose: Whether or not to print details about the pairs as they are processed
        """
        start = time.time()
        pool = ThreadPool(threads)
        # Use a starmap on the number of threads passed to do this in parallel
        self.alignments = pool.starmap(
            self.process_question_answer_pair_alignment,
            zip(self.question_id_list, self.answer_id_list, itertools.repeat(k), itertools.repeat(verbose))
        )
        # Close pool and wait for the work to finish
        pool.close()
        pool.join()
        end = time.time()
        # Print some statistics about the entire alignment data set
        print("Mean Alignment = {}".format(statistics.mean(self.alignments)))
        print("Median Alignment = {}".format(statistics.median(self.alignments)))
        print("Variance Alignment = {}".format(statistics.pvariance(self.alignments)))
        print("Standard Deviation Alignment = {}".format(statistics.pstdev(self.alignments)))
        print("Completed in {} seconds".format(end - start))

    def process_question_answer_pair_alignment(self, q_id, a_id, k=5, verbose=False, testing=False):
        """
        Given the qa pair defined by q_id and a_id use k nearest neighbours in the questions and answer spaces
        to determine the alignment of the qa pair

        :param q_id: The id of the question from the pair
        :param a_id: The id of the answer from the pair
        :param k: The number of nearest neighbours to consider for overall alignment calculations
        :param verbose: Whether or not to print details about the pairs as they are processed
        :param testing: If we are testing we do not need to worry about Numpy array conversion
        :raises ValueError: If the mapping {q_id => a_id} is not a mapping from the corpus
        :return: The alignment of the question-answer pair in the range 0-1 where higher is better
        """
        # If the given question and answer ids do not represent a pair raise a ValueError
        if self.question_to_answer_map[q_id] is not a_id:
            raise ValueError('Question ID {} does not map to answer ID {}'.format(q_id, a_id))

        q_vector = self.model.docvecs[q_id]
        a_vector = self.model.docvecs[a_id]
        if not testing:
            # NumPy arrays provide some trickiness so convert to a list now
            q_vector = q_vector.tolist()
            a_vector = a_vector.tolist()

        # Find the k nearest neighbours of Q vector
        qnn_map = self.find_k_nearest_neighbours(q_vector, self.question_vector_list, k)
        qnn_ids = []
        # The distances returned below are in ascending order, i.e 1st nearest neighbour to kth
        qnn_dists = qnn_map['distances']
        # Using those nearest neighbour vectors get the Q ids of the nearest neighbours
        for qnn_vector in qnn_map['vectors']:
            index_for_id = self.question_vector_list.index(qnn_vector)
            qnn_id = self.question_id_list[index_for_id]
            qnn_ids.append(qnn_id)

        # Determine the distance to every point in the question space
        all_q_dists = []
        for vec in self.question_vector_list:
            dist = self.calculate_euclidian_distance(q_vector, vec)
            # Remove the point itself from consideration otherwise the minimum distance will be 0
            if dist == 0:
                continue
            all_q_dists.append(dist)

        min_qnn_dist = min(all_q_dists)
        max_qnn_dist = max(all_q_dists)
        # Standardize the nearest neighbour distances within the entire question space so they are in the 0-1 range
        qnn_dists = [(qnn_dist - min_qnn_dist) / (max_qnn_dist - min_qnn_dist) for qnn_dist in qnn_dists]

        # Get the answer vectors for each of the nearest Qs, keeping the same order as the qnn_ids so the indexes
        # are analogous across arrays
        ann_ids = []
        ann_vecs = []
        ann_dists = []
        for qnn_id in qnn_ids:
            ann_id = self.question_to_answer_map[qnn_id]
            ann_ids.append(ann_id)
            ann_vec = self.model.docvecs[ann_id]
            if not testing:
                ann_vec = ann_vec.tolist()
            ann_vecs.append(ann_vec)
            # Calculate distances between A and answer vector
            ann_dists.append(self.calculate_euclidian_distance(a_vector, ann_vec))

        # Determine the distances to every point in the answer space
        all_a_dists = []
        for vec in self.answer_vector_list:
            dist = self.calculate_euclidian_distance(a_vector, vec)
            # Remove the point itself from consideration otherwise the minimum distance will be 0
            if dist == 0:
                continue
            all_a_dists.append(dist)

        min_ann_dist = min(all_a_dists)
        max_ann_dist = max(all_a_dists)

        # For each answer vector calculate it's alignment with the answer vector we are concerned with
        # currently i.e a_vector.
        ans_alignments = []
        for i in range(0, len(ann_ids)):
            # We don't need to scale the euclidean distances to 0-1 here, as this happens implicitly in the below call
            alignment = self.calculate_alignment(a_vector, ann_vecs[i], min_ann_dist, max_ann_dist)
            ans_alignments.append(alignment)

        # Calculate overall alignment of the QA pair, it will be in the range 0-1 whereby larger values indicate
        # superior alignment
        alignment_top_line = 0
        alignment_bottom_line = 0

        # Because order has been preserved through the entire process we know the nearest neighbours in the question
        # space as given by qnn_dists[x] are paired with the correct alignment given by ans_alignments[x]
        for x in range(0, len(qnn_ids)):
            alignment_top_line += (1 - qnn_dists[x]) * ans_alignments[x]
            alignment_bottom_line += (1 - qnn_dists[x])

        final_alignment = alignment_top_line / alignment_bottom_line
        if verbose:
            pair_id = "{}-q-a-{}".format(q_id, a_id)
            print("Pair {} Alignment = {}".format(pair_id, final_alignment))
        return final_alignment

    def save_model(self, model_filepath, verbose=False):
        """
        Save the Doc2Vec model to the given file path

        :param model_filepath: Where to save the model file
        :param verbose: Whether to print details of the model saving process
        """
        if verbose:
            print("Saving model using filepath: {}".format(model_filepath))
        self.model.save(model_filepath)
        if verbose:
            print("Model saving complete")

    def clean_data(self, data_array, nltk_tokenizer):
        """
        Iterate over all the text units and remove stopwords from them, returning the cleaned units

        :param data_array: The array of text units
        :param nltk_tokenizer: NTLK tokernizer object
        :return: Array of text units with stopwords removed
        """
        cleaned_data = []
        for datum in data_array:
            lower_case_datum = datum.lower()
            tokens = nltk_tokenizer.tokenize(lower_case_datum)
            cleaned_tokens = []
            # Iterate over tokens appending those that are not stop words to cleaned_tokens to preserve order
            for token in tokens:
                if token not in self.stop_words:
                    cleaned_tokens.append(token)
            cleaned_data.append(cleaned_tokens)
        return cleaned_data

    def calculate_euclidian_distance(self, vector_one, vector_two):
        """
        Calculate and return the euclidean distance between the two passed vectors

        :param vector_one: The first vector to use in the calculation
        :param vector_two: The second vector to use in the calculation
        :raises ValueError: If vectors are not of the same length a euclidean distance cannot be calculated
        :return: The euclidean distance between the two vectors
        """
        if len(vector_one) != len(vector_two):
            raise ValueError('Cannot calculate distance between two vectors of different sizes')

        summed_squared_diffs = 0
        for i in range(0, len(vector_one)):
            difference_of_components = vector_one[i] - vector_two[i]
            summed_squared_diffs += difference_of_components * difference_of_components

        return math.sqrt(summed_squared_diffs)

    def calculate_alignment(self, current_vector, examined_vector, minimum_distance, maximum_distance):
        """
        Determine the alignment of the examined_vector to the current_vector using the formula presented in the
        Dissertation

        :param current_vector: The vector to which to alignment is currently being calculated
        :param examined_vector: The vector whose relative alignment to the current vector is being calculated
        :param minimum_distance: The minimum distance observed between the current_vector and any other vector in the
                                 subspace of the passed vectors (i.e question subspace or answer subspace) with 0 not ]
                                 being considered
        :param maximum_distance: The maximum distance observed between the current_vector and any other vector in the
                                 subspace of the passed vectors (i.e question subspace or answer subspace)
        :raises ValueError: If distances are below 0, the maximum is else than the minimum, and if the distance between
                            vectors exceeds the bounds of maximum_distance and minimum_distance
        :return: The alignment which is between 0 and 1 with higher values being better or 'more aligned'
        """

        if maximum_distance < minimum_distance:
            raise ValueError('Maximum distance cannot be less than minimum distance')

        if maximum_distance < 0 or minimum_distance < 0:
            raise ValueError('Maximum or minimum distances cannot be negative')

        distance_between_vectors = self.calculate_euclidian_distance(current_vector, examined_vector)

        if distance_between_vectors < minimum_distance:
            raise ValueError('Distance calculated {} is less than the minimum distance {}'.format(
                distance_between_vectors, minimum_distance))

        if distance_between_vectors > maximum_distance:
            raise ValueError('Distance calculated {} is greater than the maximum distance {}'.format(
                distance_between_vectors, maximum_distance))

        # This next line implicitly does the 0-1 scaling of the distance for the answer vectors so there is
        # no need to scale them in the same fashion as the question vector distances, in fact scaling them before
        # hand would result in the same distance ratio
        distance_ratio = (distance_between_vectors - minimum_distance) / (maximum_distance - minimum_distance)
        return 1 - distance_ratio

    def find_k_nearest_neighbours(self, vector, all_vectors, k):
        """
        Search through all the vectors given (with the single vector removed) and determine the K nearest neighbours
        of the given single vector

        :param vector: The vector to find the K nearest neighbours for
        :param all_vectors: All of the vectors to consider in the search for the K nearest neighbours
        :param k: The number of nearest neighbours to find
        :return: A map consisting of: {'distances': array of the distances for the nearest neighbours,
                'vectors': the nearest neighbour vectors}
        """
        # Remove the given vector as we cannot consider a vector to be it's own nearest neighbour
        all_vectors_copy = list(all_vectors)
        all_vectors_copy.remove(vector)
        distances = []

        for list_vector in all_vectors_copy:
            distances.append(self.calculate_euclidian_distance(vector, list_vector))

        # Order neighbours nearest to furthest
        returned_distances = []
        returned_vectors = []

        for i in range(0, k):
            current_minimum = min(distances)
            minimum_index = distances.index(current_minimum)
            returned_distances.append(current_minimum)
            returned_vectors.append(all_vectors_copy[minimum_index])
            # Set the distance to be the max size so we dont get it again
            distances[minimum_index] = sys.maxsize

        return {'distances': returned_distances, 'vectors': returned_vectors}
