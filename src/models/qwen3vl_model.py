import copy
import json
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)

from models.base_models import ModelWrapper


AUG_LIST = (
    'random crop',
    'color inversion',
    'horizontal flip',
    'vertical flip',
    'random mask',
    'noise'
)


class Qwen3VlModel(ModelWrapper):
    def __init__(self, model_path, cd_config):
        super().__init__(cd_config, 'qwen3vl')
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map='cuda',
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.sampling_config = GenerationConfig.from_dict(
            {
                **self.base_config.to_dict(),
                'do_sample': True,
            }
        )

        self.image_mean = torch.tensor(
            [(0.48145466, 0.4578275, 0.40821073)],
            device=self.model.device
        ).view(-1, 1, 1)
        self.image_std = torch.tensor(
            [(0.26862954, 0.26130258, 0.27577711)],
            device=self.model.device
        ).view(-1, 1, 1)

    def to_pil(self, tensor):
        mean = self.image_mean.view(-1,1,1)
        std = self.image_std.view(-1,1,1)
        unnorm = torch.clamp(tensor * std + mean, 0, 1).squeeze().cpu()
        return TF.to_pil_image(unnorm)

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

        if image_path is None:                            # text-only inference
            images = None
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': query}
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt'
            ).to(self.model.device)
        else:
            images = Image.open(image_path).convert('RGB')
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': images},
                        {'type': 'text', 'text': query}
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt'
            ).to(self.model.device)
            tensor = TF.to_tensor(images)
            tensor = TF.normalize(tensor, mean=self.image_mean, std=self.image_std).unsqueeze(0).to(self.model.device)
        original_input_len = inputs['input_ids'].shape[1]
        applied_aug, reason, tensor_cd = None, None, None

        if mode == 'vcd':
            tensor_cd, applied_aug = self.apply_augmentation(aug='noise', tensor=tensor)
        elif mode == 'selfaug':
            out = self.get_self_aug(original_query)
            reason, aug = out['reason'], out['aug']
            tensor_cd, applied_aug = self.apply_augmentation(aug=aug, tensor=tensor)
        elif mode in AUG_LIST:
            tensor_cd, applied_aug = self.apply_augmentation(aug=mode, tensor=tensor)
        elif mode == 'vacode':
            with torch.inference_mode():
                logits = F.softmax(self.model(**inputs).get('logits')[:, -1, :])
                tensor_cd, max_dist = None, -1
                for aug in AUG_LIST:
                    tensor_candidate, selected_aug = self.apply_augmentation(aug=aug, tensor=tensor)
                    candidate_pv = self.processor.apply_chat_template(
                        [{'role': 'user', 'content': [
                            {'type': 'image', 'image': self.to_pil(tensor_candidate)},
                            {'type': 'text', 'text': query}
                        ]}],
                        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt'
                    ).pixel_values.to(self.model.device, dtype=self.model.dtype)

                    candidate_inputs = copy.deepcopy(inputs)
                    candidate_inputs['pixel_values'] = candidate_pv

                    score = F.softmax(self.model(**candidate_inputs).get('logits')[:, -1, :])
                    l2_norm = torch.linalg.vector_norm(logits - score, ord=2)
                    if l2_norm > max_dist:
                        max_dist = l2_norm
                        tensor_cd = tensor_candidate
                        applied_aug = selected_aug
        else:
            tensor_cd = None

        if mode in ('vcd', 'vacode', 'selfaug') + AUG_LIST:
            cd_tensor = self.processor.apply_chat_template(
                [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': self.to_pil(tensor_cd)},
                        {'type': 'text', 'text': query}
                    ]
                }],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt'
            ).pixel_values.to(self.model.device)
        else:
            cd_tensor = None

        gen_token_ids = inputs['input_ids'].clone().to(self.model.device)
        with torch.inference_mode():
            if sas:                                           # greedy decoding
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=768,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    num_beams=1,
                    top_p=1.0,
                    top_k=None
                )
                gen_token_ids = output.sequences
                if gen_token_ids.dim() > 1:
                    gen_token_ids = gen_token_ids[0]

                gen_token_ids_sliced = gen_token_ids[original_input_len:]
                outputs = self.processor.decode(gen_token_ids_sliced, skip_special_tokens=True)
                num_gen_tokens = gen_token_ids.shape[-1] - original_input_len
                applied_aug, reason = self.parse_aug(outputs).values()
                return {
                    'applied_aug': applied_aug,
                    'reason': reason,
                    'text': outputs.strip(),
                }

            else:
                output = self.model.generate(
                    **inputs,
                    cd_tensor=cd_tensor,
                    cd_config=self.cd_config,
                    max_new_tokens=768,
                    do_sample=True,
                    use_cache=True,
                    return_dict_in_generate=True,
                    num_beams=1,
                    top_p=1.0,
                    top_k=None,
                )
        gen_token_ids = output.sequences[0]

        if gen_token_ids.dim() > 1:
            gen_token_ids = gen_token_ids[0]

        gen_token_ids_sliced = gen_token_ids[original_input_len:]
        outputs_text = self.processor.decode(gen_token_ids_sliced, skip_special_tokens=True)
        num_gen_tokens = len(gen_token_ids_sliced)

        return {
            'applied_aug': applied_aug,
            'reason': reason,
            'text': outputs_text.strip(),
            'threshold': output_dict.get('threshold', None)
        }
