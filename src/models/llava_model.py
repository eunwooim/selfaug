import json
import os
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from models.base_models import ModelWrapper
from models.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.model.builder import load_pretrained_model
from models.llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from utils.utils import timer


AUG_LIST = (
    'random_crop',
    'color_inversion',
    'horizontal_flip',
    'vertical_flip',
    'random_mask',
    'noise'
)


class LlavaModel(ModelWrapper):
    def __init__(self, model_path, cd_config):
        super().__init__(cd_config, "llava")
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

        self.sampling_config = GenerationConfig.from_dict(
            {
                **self.base_config.to_dict(),
                'do_sample': True,
            }
        )

        self.image_mean = torch.tensor(
            [(0.48145466, 0.4578275, 0.40821073)],
            device=self.model.device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            [(0.26862954, 0.26130258, 0.27577711)],
            device=self.model.device
        ).view(1, 3, 1, 1)

    @timer
    def generate_sentence(
        self,
        query: str,
        append_txt: str = None,
        image_path: str = None,
        mode: str = None,
        sas: bool = False,
        oracle: tuple = None,
    ) -> dict:
        original_query = query
        if append_txt is not None:
            query = query + append_txt

        if image_path is None:                               # text-only inference
            images, image_sizes = None, None
        else:
            if self.model.config.mm_use_im_start_end:
                query = DEFAULT_IM_START_TOKEN + \
                    DEFAULT_IMAGE_TOKEN + \
                    DEFAULT_IM_END_TOKEN + '\n' + query
            else:
                query = DEFAULT_IMAGE_TOKEN + '\n' + query
            images = self.image_processor.preprocess(
                Image.open(image_path),
                return_tensors='pt'
            )['pixel_values'].to('cuda', dtype=torch.float16)

        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
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
        elif mode in AUG_LIST:
            image_tensor_cd, applied_aug = self.apply_augmentation(
                aug=mode, tensor=images
            )
        elif mode == 'vacode':
            with torch.inference_mode():
                logits = F.softmax(self.model.generate(
                    input_ids,
                    images=images,
                    output_scores=True,
                    generation_config=self.sampling_config,
                ).scores[0])
                image_tensor_cd, max_dist = None, -1
                for aug in AUG_LIST:
                    tensor_cd, selected_aug = self.apply_augmentation(
                        aug=aug, tensor=images
                    )
                    score = F.softmax(self.model.forward(
                        input_ids,
                        images=tensor_cd,
                    ).get('logits')[:,-1,:])
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
                    input_ids,
                    images=images,
                    images_cd=image_tensor_cd,
                    generation_config=self.sampling_config,
                    **self.cd_param
                )
            else:                                             # greedy decoding
                output_dict = self.model.generate(
                    input_ids,
                    generation_config=self.greedy_config,
                )
        input_token_len = input_ids.shape[1]
        output_token_len = output_dict.sequences.shape[1]
        n_diff_input_output = (input_ids != output_dict.sequences[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_dict.sequences[:, input_token_len:], skip_special_tokens=True)[0]
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return {
            'applied_aug': applied_aug,
            'reason': reason,
            'text': outputs.strip(),
            'threshold': output_dict.get('threshold', None)
        }
