from RhetoricAnalysis.AlignmentResearch.code import QAPairAlignmentCalculator
import json
import datetime

pms = ["thatcher", "major", "blair", "brown", "cameron", "may"]
parties = ["conservative", "conservative", "labour", "labour", "conservative", "conservative"]


if __name__ == '__main__':
    for i in reversed(range(0, len(pms))):

        qa_pair_array = []
        question_map = {}
        answer_map = {}

        with open('cleaned_parliament.json') as parliament_file:
            parliament_object = json.load(parliament_file)
            parliament_file.close()

        for question_or_answer_object in parliament_object:
            if question_or_answer_object['is_answer']:
                answer_map[question_or_answer_object['reply-to']] = question_or_answer_object
            else:
                question_map[question_or_answer_object['id']] = question_or_answer_object

        for qid in question_map.keys():
            question_object = question_map[qid]
            answer_object = answer_map[qid]
            pair_map = {'govt': answer_object['govt'], 'date': answer_object['date'],
                        'question': question_object, 'answer': answer_object}
            qa_pair_array.append(pair_map)

        # Sort Chronologically
        qa_pair_array.sort(key=lambda item: datetime.datetime.strptime(item['date'], "%Y-%m-%d"))
        # Write all those Questions and answers for this govt into an array to be written to file as the corpus
        qa_array_with_only_pm_included = []
        pairs_retained = 0
        for pair in qa_pair_array:
            if pair['govt'] == pms[i] and pair['answer']['user-info']['party'] == parties[i]:
                pairs_retained += 1
                qa_array_with_only_pm_included.append(pair['question'])
                qa_array_with_only_pm_included.append(pair['answer'])

        print("Pairs Retained = {}".format(pairs_retained))
        print("Including only '{}' pairs from dataset complete, writing file and building matrix".format(pms[i]))

        corpus_filepath = '../resources/pm_files_{}_only_included.json'.format(pms[i])

        with open(corpus_filepath, "w") as file:
            file.write(json.dumps(qa_array_with_only_pm_included, indent=4))
            file.close()

        print("File written")

        qa_alignment = QAPairAlignmentCalculator(
            corpus_filepath=corpus_filepath,
            model_filepath='../resources/parliament_doc2vec_2.model',
            stop_words_filepath='../resources/stop_words.txt',
            verbose=True
        )

        qa_alignment.process_all_pair_alignments(k=5, threads=8, verbose=False)
        qa_alignment.save_all_pair_alignments_to_file('pm_files_{}_only_included_alignments.csv'.format(pms[i]))