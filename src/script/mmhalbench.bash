#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

model_id="llava15_7b"     # "llava15_7b" "qwenvl", "instructblip" "qwen3vl_8b"
cd_mode="selfaug"         # "no_vcd" "vcd" "vacode" "selfaug"
cd_alpha=1
cd_beta=0.1
cd_tau=0.5
crop_ratio=2.0
mask_ratio=2.0
noise_step=500
seeds=(11 21 31 41 51)
image_folder=/path/to/your/MMHal-Bench/images
cuda=0

for seed in "${seeds[@]}"; do
  answer_file=./output/${cd_mode}/mmhalbench/${model_id}_seed${seed}.jsonl

  python eval/mmhalbench.py \
    --model-id ${model_id} \
    --image-folder ${image_folder} \
    --question-file ./data/MMHalBench/mmhalbench.jsonl \
    --answers-file ${answer_file} \
    --crop-ratio ${crop_ratio} \
    --mask-ratio ${mask_ratio} \
    --noise-step ${noise_step} \
    --cd-mode ${cd_mode} \
    --cd-alpha ${cd_alpha} \
    --cd-beta ${cd_beta} \
    --seed ${seed} \
    --cuda ${cuda} \
    --cd-tau ${cd_tau}

  python eval/eval_mmhalbench.py \
    --model-outputs ${answer_file}
done
