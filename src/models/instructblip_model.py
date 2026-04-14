import os
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import GenerationConfig
import numpy as np

from models.base_models import ModelWrapper
from lavis.models import load_model_and_preprocess


VACODE_LIST = (
    'random_crop',
    'color_inversion',
    'horizontal_flip',
    'vertical_flip',
    'random_mask',
    'noise'
)


class InstructBlipModel(ModelWrapper):
    def __init__(self, model_path, cd_config):
        super().__init__(cd_config, 'instructblip')
        self.cd_config = cd_config
        model, image_processor, _ = load_model_and_preprocess(
            name='blip2_vicuna_instruct',
            model_type='vicuna7b',
            is_eval=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = model
        self.image_processor = image_processor['eval']

        self.image_mean = torch.tensor(
            [(0.48145466, 0.4578275, 0.40821073)],
            device=self.device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            [(0.26862954, 0.26130258, 0.27577711)],
            device=self.device
        ).view(1, 3, 1, 1)

    def generate_sentence(
        self,
        query: str,
        append_txt: str = None,
        image_path: str = None,
        mode: str = None,
        sas: bool = False
    ) -> dict:
        original_query = query
        if append_txt is not None:
            query = query + append_txt
        input_ids = {'prompt': query}

        if image_path is None:                               # text-only inference
            images, image_sizes = None, None
        else:
            images = self.image_processor(
                Image.open(image_path).convert('RGB')
            ).unsqueeze(0).to('cuda')

        applied_aug, reason = None, None
        if mode == 'vcd':
            image_tensor_cd, applied_aug = self.apply_augmentation(
                aug='noise', tensor=images
            )
        elif mode == 'selfaug':
            out = self.get_self_aug(original_query)
            reason, aug = out['reason'], out['aug']
            image_tensor_cd, applied_aug = self.apply_augmentation(
                aug=aug, tensor=images
            )
        elif mode in VACODE_LIST:
            image_tensor_cd, applied_aug = self.apply_augmentation(
                aug=mode, tensor=images
            )
        elif mode in 'vacode':
            with torch.inference_mode():
                logits = F.softmax(self.model.generate(
                    {'image': images, **input_ids},
                    use_nucleus_sampling=True,
                    num_beams=1,
                    top_p=1,
                    repetition_penalty=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                ).scores[0])
                image_tensor_cd, max_dist = None, -1
                for aug in VACODE_LIST:
                    tensor_cd, selected_aug = self.apply_augmentation(
                        aug=aug, tensor=images
                    )
                    score = F.softmax(self.model.generate(
                        {'image': tensor_cd, **input_ids},
                        use_nucleus_sampling=True,
                        num_beams=1,
                        top_p=1,
                        repetition_penalty=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                    ).scores[0])
                    l2_norm = torch.linalg.vector_norm(logits - score, ord=2)
                    if l2_norm > max_dist:
                        max_dist = l2_norm
                        image_tensor_cd = tensor_cd
                        applied_aug = selected_aug
        else:
            image_tensor_cd = None

        with torch.inference_mode():
            if not sas:
                output_dict = self.model.generate(
                    {'image': images, **input_ids},
                    images_cd=image_tensor_cd,
                    use_nucleus_sampling=True,
                    num_beams=1,
                    top_p=1,
                    repetition_penalty=1,
                    return_dict_in_generate=True,
                    **self.cd_param
                )
            else:                                             # greedy decoding
                output_dict = self.model.generate(
                    input_ids,
                    use_nucleus_sampling=False,
                    repetition_penalty=1.1,
                    return_dict_in_generate=True,
                )
        outputs = output_dict.sequences.clone()
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        outputs = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return {
            'applied_aug': applied_aug,
            'reason': reason,
            'text': outputs.strip(),
            'threshold': output_dict.get('threshold', None)
        }
