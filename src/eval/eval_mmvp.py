import argparse
import json
import pandas as pd
import re


def is_correct(answer, options, gt):
    correct_choice_full = ''
    correct_choice_text = ''
    gt_bare = gt.strip('() ')
    choices = re.findall(r'\([a-z]\)\s[^()]+', options)

    for choice in choices:
        if choice.startswith(gt):
            correct_choice_full = choice.strip()
            correct_choice_text = choice.split(')', 1)[1].strip()
            break

    if not correct_choice_text:
        return False

    answer = answer.strip().lower()
    return (
        answer == gt.lower() or
        answer == gt_bare.lower() or
        answer == correct_choice_text.lower() or
        answer == correct_choice_full.lower()
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-outputs', type=str, required=True)
    parser.add_argument('--gt-dir', type=str, default='./data/MMVP/Questions.csv')
    return parser.parse_args()


def main(args):
    gt_files = {i: j for i, j in pd.read_csv(args.gt_dir).iterrows()}
    gen_files = [json.loads(i) for i in open(args.model_outputs, 'r')]

    correct, index, rounds, total = 0, 0, 0, 0
    for i, line in enumerate(gen_files):
        idx, answer = line['question_id'], line['text']
        gt, options = gt_files[idx]['Correct Answer'], gt_files[idx]['Options']

        index += 1
        if is_correct(answer, options, gt):
            rounds += 1
        if index == 2:
            if rounds == 2:
                correct += 1
            total += 1
            index, rounds = 0, 0

    print(f'Accuracy: {100 * correct / total}')
    return


if __name__ == '__main__':
    args = get_args()
    main(args)
