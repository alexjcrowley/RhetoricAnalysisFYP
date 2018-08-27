import json
import datetime
import statistics
from RhetoricAnalysis.FragmentCooccurrenceMatrixResearch.code.fragment_cooccurrence_matrix import FragmentCooccurrenceMatrix

answer_arc_objects = {}
question_arc_objects = {}

pms = ["thatcher", "major", "blair", "brown", "cameron", "may"]
parties = ["conservative", "conservative", "labour", "labour", "conservative", "conservative"]

if __name__ == '__main__':
    for i in range(0, len(pms)):
        qa_pair_array = []
        question_map = {}
        answer_map = {}

        with open('../../datasets/cleaned_parliament.json') as parliament_file:
            parliament_object = json.load(parliament_file)
            parliament_file.close()

        for question_or_answer_object in parliament_object:
            if question_or_answer_object['is_answer']:
                answer_map[question_or_answer_object['reply-to']] = question_or_answer_object
            else:
                question_map[question_or_answer_object['id']] = question_or_answer_object

        for id in question_map.keys():
            question_object = question_map[id]
            answer_object = answer_map[id]
            pair_map = {'govt': answer_object['govt'], 'date': answer_object['date'],
                        'question': question_object, 'answer': answer_object}
            qa_pair_array.append(pair_map)

        # Sort Chronologically
        qa_pair_array.sort(key=lambda item: datetime.datetime.strptime(item['date'], "%Y-%m-%d"))

        # Write all those Questions and answers not for this govt into an array to be written to file and matrix built
        qa_array_with_pm_excluded = []
        removals = 0
        for pair in qa_pair_array:
            if pair['govt'] != pms[i]:
                qa_array_with_pm_excluded.append(pair['question'])
                qa_array_with_pm_excluded.append(pair['answer'])
            else:
                removals += 1

        print("Pair Removals = {}".format(removals))
        print("Removing {} pairs from dataset complete, writing file and building matrix".format(pms[i]))

        with open('excluded_pm_files_{}.json'.format(pms[i]), "w") as file:
            file.write(json.dumps(qa_array_with_pm_excluded, indent=4))
            file.close()

        print("File written")

        # Build our fragment matrix
        fragment_matrix = FragmentCooccurrenceMatrix.from_full_ccat_run(
            dataset_name='excluded_pm_files_{}'.format(pms[i]),
            dataset_file_extension='.json',
            minimum_fragment_occurrence_frequency=200,
            verbose=True,
            stop_words_filename='../resources/stop_words.txt',
            remove_single_word_fragments=False,
            num_clusters=8,
            random_seed=125
        )

        # Build the answer fragment arrays for each pair_idx
        with open('testing-resource-files/answer_arcs.json') as answer_arcs:
            for answer_line in answer_arcs:
                answer_object = json.loads(answer_line)
                # The next 3 lines remove the 'span...' ending to the pair_idx as we want to consider entire q-a pairs
                pair_idx = answer_object['pair_idx']
                index = pair_idx.index('span')
                pair_idx = pair_idx[:index]

                # If we haven't seen this pair_idx before make the array for it
                if answer_arc_objects.get(pair_idx, None) is None:
                    answer_arc_objects[pair_idx] = []

                for answer_fragment in answer_object['arcs']:
                    answer_arc_objects[pair_idx].append(answer_fragment)

            answer_arcs.close()

        # Build the question fragment arrays for each pair_idx
        with open('testing-resource-files/question_arcs.json') as question_arcs:
            for question_line in question_arcs:
                question_object = json.loads(question_line)
                # The next 3 lines remove the 'span...' ending to the pair_idx as we want to consider entire q-a pairs
                pair_idx = question_object['pair_idx']
                index = pair_idx.index('span')
                pair_idx = pair_idx[:index]

                # If we haven't seen this pair_idx before make the array for it
                if question_arc_objects.get(pair_idx, None) is None:
                    question_arc_objects[pair_idx] = []

                for question_fragment in question_object['arcs']:
                    question_arc_objects[pair_idx].append(question_fragment)

            question_arcs.close()

        rank_factors = []
        for qa_pair in qa_pair_array:
            govt = qa_pair['govt']

            # Now we only want to analyse those question-answer pairs actually from the desired govt
            # The party check means we do not given the PM credit when a coalition member answers
            if govt != pms[i] or qa_pair['answer']['user-info']['party'] != parties[i]:
                continue

            question_text = qa_pair['question']['text']
            answer_text = qa_pair['answer']['text']
            pair_idx = "{}-q-a-{}".format(qa_pair['question']['id'], qa_pair['answer']['id'])

            try:
                question_fragments = question_arc_objects[pair_idx]
                answer_fragments = answer_arc_objects[pair_idx]
            except KeyError:
                # In some cases we have pair_idx's that didn't appear in answer_arcs or question_arcs,
                # I assume this is because of bad data
                continue

            # If either question or answer fragments are empty then we cant do anything with the QA pair
            if len(question_fragments) == 0 or len(answer_fragments) == 0:
                continue

            # Get the summed co occurrences for those question fragments that were in the matrix
            answer_fragment_rankings = \
                fragment_matrix.get_summed_cooccurrence_frequencies_for_fragment_list(question_fragments,
                                                                                      True)['in_matrix']

            # If none of the question fragments were in the matrix we skip to next QA pair
            if len(answer_fragment_rankings) == 0:
                continue

            sorted_ranks = list(answer_fragment_rankings)
            # Sort in descending order so most co-occurring are best ranked
            sorted_ranks.sort(reverse=True)

            number_of_answer_frags = len(answer_fragments)
            total_rank = 0

            # For each answer fragment find the index of the answer fragment (or skip this fragment if it isn't in
            # fragment_matrix, and decrement the number of answer frags for this QA pair by 1) ... then using the index
            # find the co occurrence value for that fragment, then find the value in the sorted ranks and add the index,
            # which is the rank (where the possible ranks are 0 to fragment_matrix.get_number_of_answer_fragments() - 1)
            # to the total rank for this answer
            for answer_fragment in answer_fragments:
                index_of_value = fragment_matrix.get_answer_fragment_id(answer_fragment)
                if index_of_value is None:
                    number_of_answer_frags -= 1
                    continue

                value = answer_fragment_rankings[index_of_value]
                rank = sorted_ranks.index(value)
                total_rank += rank

            # If we ended up having no answer fragments in the matrix then skip this QA pair
            if number_of_answer_frags == 0:
                continue

            # Ranking factor finds the fraction of the maximum possible rank that this answer has and takes it away from
            # 1, where subtracting from 1 means better answers have higher values (since the best rank is 0, so 'high'
            # ranked answers fragments have low actual rank values before the subtraction)
            rank_factor = 1 - (total_rank / ((fragment_matrix.get_number_of_answer_fragments() - 1) * number_of_answer_frags))
            rank_factors.append(rank_factor)

        # Print out some details about the current pair and increment the total number processed
        print("Mean RF = {}".format(statistics.mean(rank_factors)))
        print("Median RF = {}".format(statistics.median(rank_factors)))
        print("Max RF = {}".format(max(rank_factors)))
        print("Min RF = {}".format(min(rank_factors)))
        print("Num Examined = {}".format(len(rank_factors)))
