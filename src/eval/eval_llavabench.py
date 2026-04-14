import argparse
from collections import defaultdict
import json
import os
import time

import numpy as np
from openai import OpenAI
from tqdm import tqdm


NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(client, content, max_tokens=1024):
    while True:
        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                    },
                    {
                        'role': 'user',
                        'content': content,
                    }
                ],
                temperature=0.0,
                seed=42,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def main(args):
    api_key="your-api-key"
    client = OpenAI(api_key=api_key)


    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.gpt_answer))
    f_ans2 = open(os.path.expanduser(args.model_outputs))
    rule_dict = json.load(open(os.path.expanduser(args.rule)))

    if os.path.isfile(os.path.expanduser(args.model_outputs)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.model_outputs))]
    else:
        cur_reviews = []
    
    review_file = open(args.model_outputs, 'a')
    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    idx, judge_results, total_score = 0, {}, defaultdict(list)
    judge_result_dir = os.path.splitext(args.model_outputs)[0]
    judge_result_path = os.path.join(judge_result_dir, 'results.json')
    score_result_path = os.path.join(judge_result_dir, 'scores.json')
    os.makedirs(judge_result_dir, exist_ok=True)

    for ques_js, ans1_js, ans2_js in tqdm(zip(f_q, f_ans1, f_ans2), total=60, ncols=79):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]
        qid = inst['id']

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f'Visual QA category not found in rule file: {category}.'
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}')

        review = get_eval(client, content, 1024)
        scores = parse_score(review)

        total_score[category].append(scores)
        total_score['all'].append(scores)
        judge_results[qid] = {'review': review, 'scores': scores}

    result_dict = defaultdict(list)
    print(f'Results for {args.model_outputs}:')
    for k, v in sorted(total_score.items()):
        stats = np.asarray(v).mean(0).tolist()
        result_dict[k] = [round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1)]
        print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
    print('\n\n')

    with open(judge_result_path, 'w') as f:
        json.dump(judge_results, f, indent=4)
    with open(score_result_path, 'w') as f:
        json.dump(total_score, f, indent=4)
    return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-outputs", type=str)
    parser.add_argument("--question", type=str, default='./data/llavabench/llavabench.jsonl')
    parser.add_argument("--context", type=str, default='./data/llavabench/context.jsonl')
    parser.add_argument("--gpt-answer", type=str, default='./data/llavabench/answers_gpt4.jsonl')
    parser.add_argument("--rule", type=str, default='./data/llavabench/rule.json')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
