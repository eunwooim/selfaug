# [ICLR 2026] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models
Authors: [Eun Woo Im](https://eunwooim.github.io), [Muhammad Kashif Ali](https://github.com/MKashifAli), [Vivek Gupta](https://coral-lab-asu.github.io).

This repository contains the official implementation of Self-Aug, a training-free decoding method for mitigating hallucinations in large vision-language models (LVLMs).
Self-Aug emphasizes query-augmentation alignment and entropy utilization improve generation reliability.
More information are available at the links below.

[[📄 arXiv](https://arxiv.org/abs/2510.13315)] [[🌐 Project Page](https://eunwooim.github.io/selfaug)]

## Updates
- [04/14/2026] 🚀 Code for Self-Aug released
- [01/26/2026] 🥳 Accepted to **ICLR 2026** with updated title
- [10/15/2025] 🚨 Self-Augmented Visual Contrastive Decoding is now available on arXiv

## 🌴 Repo structure
```
.
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── MMVP
│   │   ├── POPE
│   │   └── ...
│   ├── eval
│   │   ├── eval_mme.py
│   │   ├── eval_mmvp.py
│   │   ├── mme.py
│   │   ├── mmvp.py
│   │   └── ...
│   ├── models
│   │   ├── Qwen_VL
│   │   ├── lavis
│   │   ├── llava
│   │   ├── __init__.py
│   │   ├── base_models.py
│   │   ├── llava_model.py
│   │   └── ...
│   ├── script
│   │   ├── mme.bash
│   │   ├── mmvp.bash
│   │   ├── pope.bash
│   │   └── ...
│   └── utils
│       ├── __init__.py
│       ├── utils.py
│       └── vcd_sample.py
├── README.md
├── requirements.txt
└── ...
```

## ⚙️ Usage

### 🌏 Environment setup

1. For LLaVA-1.5, Qwen-VL, InstructBLIP, follow the command below.
```bash
conda create -f environment2.yml
conda activate selfaug
```
2. For Qwen3-VL, follow the command below.
```
conda create -f environment2.yml
conda activate selfaug2
```
3. ⚠️ If installation fails due to dependency conflicts, please refer to the [official VCD repository](https://github.com/DAMO-NLP-SG/VCD), as this implementation builds on top of it.

### ✅ Prepare checkpoints
Download model checkpoints and place them as follows.
```
./src/models
├── llava
│   ├── llava-1.5-7b                          # add the checkpoint here
│   |   ├── config.json
│   |   ├── pytorch_model-00001-of-00002.bin
│   |   ├── ...
│   |   ├── tokenizer.model
│   |   └── tokenizer_config.json
│   ├── llava-1.5-13b                         # add the checkpoint here
│   └── ...
├── lavis
│   ├── instruct_blip_vicuna7b_trimmed.pth    # add the checkpoint here
│   ├── vicuna-7b-v1.1                        # add the checkpoint here
│   |   ├── config.json
│   |   └── ...
│   └── ...
└── llava_model.py
```
You can download the checkpoints from:
- LLaVA-1.5: [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main), [LLaVA-1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b/tree/main)
- InstructBLIP: [instruct_blip_vicuna7b_trimmed.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth"), [vicuna-7B-v1.1](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)

Note: For InstructBLIP, you should also modify the path to the checkpoint (instruct_blip_vicuna7b_trimmed.pth) in [config file](./src/models/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml).

### ⬇️ Download Datasets
- MME: [pairs](https://huggingface.co/datasets/darkyarding/MME/blob/main/MME_Benchmark_release_version.zip)
- POPE: [images](http://images.cocodataset.org/zips/val2014.zip) for both MS-COCO and A-OKVQA
- MMVP: [images](https://huggingface.co/datasets/MMVP/MMVP/tree/main)
- MM-Vet: [images](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip)
- LLaVA-in-the-Wild: [images](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main)
- MMHal-Bench: [images](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/tree/main)

### 🔑 API Key Settings
For LLM-as-a-Judge benchmarks (e.g., LLaVA-Bench, MMHal-Bench, MM-Vet), add your API keys in `./src/eval/{benchmark-name}.py` inside the corresponding evaluation scripts.

### 🏃 How to run
Run a sample evaluation:
```
cd selfaug/src
bash script/mme.bash
bash script/mmvp.bash
...
```

## 📎 Citation
If you find this work useful, please cite:
```
@article{im2025self,
  title={Self-Augmented Visual Contrastive Decoding},
  author={Im, Eun Woo and Ali, Muhammad Kashif and Gupta, Vivek},
  journal={arXiv preprint arXiv:2510.13315},
  year={2025}
}
```

# Acknowledgements
- The code in this repository is heavily adopted from [VCD](https://github.com/DAMO-NLP-SG/VCD) and [CODE](https://github.com/IVY-LVLM/CODE), we thank the authors for opensourcing their work.

- ⚠️ In order to improve readability, this repository have significantly rewritten from the original repository. If you encounter any bugs or issues, please feel free to raise issues.
