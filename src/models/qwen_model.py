import os
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig

from models.base_models import ModelWrapper
from models.Qwen_VL.modeling_qwen import QWenLMHeadModel
from utils.utils import timer


VACODE_LIST = (
    'random_crop',
    'color_inversion',
    'horizontal_flip',
    'vertical_flip',
    'random_mask',
    'noise'
)


class QwenVlModel(ModelWrapper):
    def __init__(self, model_path, cd_config):
        super().__init__(cd_config, 'qwenvl')
        self.cd_config = cd_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen-VL',
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.model = QWenLMHeadModel.from_pretrained(
            'Qwen/Qwen-VL',
            device_map='cuda' if torch.cuda.is_available() else 'cpu',
            bf16=True,
        ).eval()

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
        image_path: Image.Image = None,
        mode: str = None,
        sas: bool = False
    ) -> str:
        original_query = query
        if append_txt is not None:
            query = query + append_txt

        if image_path is None:                               # text-only inference
            images, image_sizes = None, None
        else:
            query = '<img>{}</img>{} Answer:'.format(image_path, query)
            pil_img = Image.open(image_path).convert('RGB')
            images = self.model.transformer.visual.image_transform(pil_img).unsqueeze(0).cuda()

        input_ids = self.tokenizer(
            [query],
            return_tensors='pt',
            padding='longest'
        ).to('cuda')

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
                logits = F.softmax(self.model.forward(
                    input_ids.input_ids,
                    images=images,
                ).get('logits')[:,-1,:])
                image_tensor_cd, max_dist = None, -1
                for aug in VACODE_LIST:
                    tensor_cd, selected_aug = self.apply_augmentation(
                        aug=aug, tensor=images
                    )
                    score = F.softmax(self.model.forward(
                        input_ids.input_ids,
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
                    input_ids=input_ids.input_ids,
                    attention_mask=input_ids.attention_mask,
                    do_sample=True,
                    max_new_tokens=2048,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eod_id,
                    eos_token_id=self.tokenizer.eod_id,
                    temperature=1.0,
                    top_p=1,
                    top_k=None,
                    images=images,
                    images_cd=image_tensor_cd,
                    cd_alpha=self.cd_config.cd_alpha,
                    cd_beta=self.cd_config.cd_beta,
                    cd_tau=self.cd_config.cd_tau,
                    return_dict_in_generate=True,
                )
            else:                                             # greedy decoding
                output_dict = self.model.generate(
                    input_ids=input_ids.input_ids,
                    images=images,
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True,
                    repetition_penalty=1.1,
                    return_dict_in_generate=True
                )
        outputs = [
            self.tokenizer.decode(
                out[input_ids.input_ids.size(1):].cpu(),
                skip_special_tokens=True
            ).strip() for out in output_dict.sequences
        ][0].strip()
        return {
            'applied_aug': applied_aug,
            'reason': reason,
            'text': outputs,
            'threshold': output_dict.get('threshold', None)
        }