import os
import json
import argparse
from tqdm import tqdm


COCO = {
    'adversarial': './data/POPE/coco/coco_pope_adversarial.json',
    'popular': './data/POPE/coco/coco_pope_popular.json',
    'random': './data/POPE/coco/coco_pope_random.json',
}
AOKVQA = {
    'adversarial': './data/POPE/aokvqa/aokvqa_pope_adversarial.json',
    'popular': './data/POPE/aokvqa/aokvqa_pope_popular.json',
    'random': './data/POPE/aokvqa/aokvqa_pope_random.json',
}

parser = argparse.ArgumentParser()
parser.add_argument('--gt-files', type=str, default=None)
parser.add_argument('--model-outputs', type=str, default='')
args = parser.parse_args()

# open ground truth answers
if args.gt_files is None:
    if 'coco' in args.model_outputs:
        data_type = COCO
    else:
        data_type = AOKVQA
    if 'adversarial' in args.model_outputs:
        args.gt_files = data_type['adversarial']
    elif 'popular' in args.model_outputs:
        args.gt_files = data_type['popular']
    else:
        args.gt_files = data_type['random']
print(args.gt_files)
gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), 'r')]

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.model_outputs), 'r')]

# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0

# compare answers
for index, line in enumerate(gt_files):
    idx = line['question_id']
    gt_answer = line['label']
    assert idx == gen_files[index]['question_id']
    gen_answer = gen_files[index]['text']
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
# calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions
# report results
print(f'Accuracy: {accuracy*100}')
print(f'F1: {f1*100}\n')
