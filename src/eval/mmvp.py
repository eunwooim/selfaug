import argparse
import json
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from models.base_models import load_model
from utils.utils import (
    VcdConfig,
    get_last_qid,
    set_seed,
    setup_path
)
setup_path()


def eval_model(args):
    # Data and result file loading
    questions = [i for i in pd.read_csv(args.question_file).iterrows()]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Print configs
    print('='*79)
    for key, value in vars(args).items():
        print(f'{key:<20}: {value}')
    print('='*79)

    # Model loading
    cd_config = VcdConfig(
        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        cd_tau=args.cd_tau,
        crop_ratio=args.crop_ratio,
        mask_ratio=args.mask_ratio,
        noise_step=args.noise_step,
        cd_mode=args.cd_mode,
    )
    model = load_model(args.model_id, cd_config)

    # Can save computation by recording SAS results
    if args.sas_path is not None:
        sas = [json.loads(i) for i in open(os.path.expanduser(args.sas_path), 'r')]
    else:
        sas = None

    # Open result file and loop
    ans_file = open(answers_file, 'w')
    for idx, row in tqdm(questions, ncols=79):
        image_file, query = f"{row['lndex']}.jpg", f"{row['Question']} {row['Options']}"
        outputs = model.generate_sentence(
            query=query,
            image_path=os.path.join(args.image_folder, image_file),
            append_txt="\nAnswer with the option's letter from the given choices directly.",
            mode=sas[idx]['applied_aug'] if cd_config.cd_mode == 'selfaug' else cd_config.cd_mode,
        )

        ans_file.write(json.dumps({'question_id': idx,
                                   'prompt': query,
                                   'text': outputs.get('text'),
                                   'applied_aug': outputs.get('applied_aug'),
                                   'reason': outputs.get('reason'),
                                   'thresholds': outputs.get('thresholds'),
                                   'model_id': args.model_id,
                                   'image': image_file,
                                   'runtime': outputs.get('runtime'),
                                   'metadata': {}}) + '\n')
        ans_file.flush()
    ans_file.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=str, default='llava15_7b')
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--question-file', type=str, default='tables/question.jsonl')
    parser.add_argument('--answers-file', type=str, default='answer.jsonl')
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    # generation config
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--top_k', type=int, default=None)
    # VCD configs
    parser.add_argument('--crop-ratio', type=float, default=2.0)
    parser.add_argument('--mask-ratio', type=float, default=2.0)
    parser.add_argument('--noise-step', type=int, default=500)
    parser.add_argument('--cd-mode', type=str, default='no_vcd')
    parser.add_argument('--cd-alpha', type=float, default=1)
    parser.add_argument('--cd-beta', type=float, default=0.1)
    parser.add_argument('--cd-tau', type=float, default=None)
    parser.add_argument('--sas-path', type=str, default=None)
    # others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=str, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    set_seed(args.seed)
    eval_model(args)
