import random
import warnings

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from transformers import GenerationConfig

from utils.utils import disable_torch_init
from utils.vcd_sample import evolve_vcd_sampling
from utils.vcd_sample2 import evolve_vcd_sampling2


warnings.filterwarnings('ignore')
SUPPORTED_MODELS = {
    'llava15_7b': './models/llava/llava-v1.5-7b',
    'llava15_13b': './models/llava/llava-v1.5-13b',
    'instructblip': './models/lavis/vicuna-7b-v1.1',
    'qwenvl': None,
    'qwen3vl_8b': 'Qwen/Qwen3-VL-8B-Instruct',
    'qwen3vl_32b': 'Qwen/Qwen3-VL-32B-Instruct',
}


class ModelWrapper:
    def __init__(self, cd_config, which_model):
        if 'qwen3vl' in which_model:
            evolve_vcd_sampling2()
        else:
            evolve_vcd_sampling()
        self.cd_config = cd_config

        # generation config setup
        self.base_config = GenerationConfig(
            max_new_tokens=768,
            temperature=1.0,
            top_p=1,
            top_k=None,
            use_cache=True,
            return_dict_in_generate=True,
            cd_alpha = cd_config.cd_alpha
        )
        self.greedy_config = GenerationConfig.from_dict(
            {
                **self.base_config.to_dict(),
                'do_sample': False,
                'num_beams': 1,
            }
        )
        self.cd_param = {
            'cd_alpha': cd_config.cd_alpha,
            'cd_beta': cd_config.cd_beta,
            'cd_tau': cd_config.cd_tau,
        }

    def __getattr__(self, name):
        return getattr(self.model, name)

    def add_diffusion_noise(self, image_tensor, noise_step):
        num_steps = 1000  # Number of diffusion steps

        # decide beta in each step
        betas = torch.linspace(-6,6,num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

        # decide alphas in each step
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        def q_x(x_0,t):
            noise = torch.randn_like(x_0)
            alphas_t = alphas_bar_sqrt[t]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
            return (alphas_t * x_0 + alphas_1_m_t * noise)

        noise_delta = int(noise_step) # from 0-999
        noisy_image = image_tensor.clone()
        image_tensor_cd = q_x(noisy_image,noise_step) 

        return image_tensor_cd

    def apply_augmentation(
        self, aug, tensor, crop_ratio=2.5, mask_ratio=2, noise_step=500
    ) -> torch.Tensor:
        selected_aug = None
        if aug is None:
            return (tensor, None)
        elif 'crop' in aug:
            _, _, h, w = tensor.shape
            crop_size = int(min(h, w) // crop_ratio)
            cd_tensor = T.Compose([
                T.RandomCrop(size=(crop_size, crop_size)),
                T.Resize(size=(h, w), antialias=True)
            ])(tensor)
            selected_aug = 'random_crop'
        elif 'horizontal' in aug:
            cd_tensor = F.hflip(tensor)
            selected_aug = 'horizontal_flip'
        elif 'vertical' in aug:
            cd_tensor = F.vflip(tensor)
            selected_aug = 'vertical_flip'
        elif 'inver' in aug:
            cd_tensor = self.invert_color(tensor)
            selected_aug = 'color_inversion'
        elif 'mask' in aug:
            _, _, h, w = tensor.shape
            mask_size = int(min(w,h) // mask_ratio)
            top = random.randint(0, h - mask_size)
            left = random.randint(0, w - mask_size)
            cd_tensor = tensor.clone()
            cd_tensor[:, :, top:top+mask_size, left:left+mask_size] = 0.
            selected_aug = 'random_mask'
        else:
            cd_tensor = self.add_diffusion_noise(tensor, noise_step)
            selected_aug = 'noise'
        return (cd_tensor.to('cuda', dtype=torch.float16), selected_aug)

    def invert_color(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # input img_tensor must be a shape of [1, 3, H, W]
        tensor = torch.clamp(img_tensor * self.image_std + self.image_mean, 0, 1)
        return (F.invert(tensor) - self.image_mean) / self.image_std

    def get_self_aug(self, query):
        prompt = self.self_aug_prompt(query)
        output = self.generate_sentence(query=prompt, mode='no_vcd', sas=True)['text']
        return self.parse_aug(output)

    def parse_aug(self, result):
        result = result.lower()
        try:
            aug = result.split('choice:')[1].strip()
        except:
            aug = result
        return {
            'aug': aug,
            'reason': result
        }

    def self_aug_prompt(self, text, reason=True, icl=True):
        if reason and icl:
            return PROMPT_FULL.format(text)
        elif (not reason) and icl:
            return PROMPT_ICL.format(text)
        elif reason and (not icl):
            return PROMPT_REASON.format(text)
        else:
            return PROMPT_BASE.format(text)


def load_model(model_id: str, cd_config):
    disable_torch_init()
    assert model_id in SUPPORTED_MODELS, f"{model_id} not supported."
    model_path = SUPPORTED_MODELS[model_id]
    if 'llava15' in model_id:
        from models.llava_model import LlavaModel
        model = LlavaModel(model_path, cd_config)
    elif model_id == 'qwenvl':
        from models.qwen_model import QwenVlModel
        model = QwenVlModel(model_path, cd_config)
    elif model_id == 'instructblip':
        from models.instructblip_model import InstructBlipModel
        model = InstructBlipModel(model_path, cd_config)
    else:
        from models.qwen3vl_model import Qwen3VlModel
        model = Qwen3VlModel(model_path, cd_config)
    return model


PROMPT_FULL = """You are an expert data augmentation analyst. Your task is to select the single most semantically disruptive image augmentation that most effectively invalidates the question's premise or prevents a confident answer. Provide a clear reason explaining why the augmentation is chosen, then state your final choice.

## Augmentations and Their Effects ##
- Vertical flip: Flips image top-to-bottom. Disrupts questions about “above”, “below”, “under” or reading orientation.
- Color inversion: Replaces each color with its complement. Disrupts questions relying on accurate color identification.
- Random crop: Removes random parts of the image. Disrupts questions requiring global context or peripheral objects.
- Random mask: Occludes portions of the image. Disrupts object presence, count, or attribute recognition.
- Noise: Adds visual distortion. Disrupts questions requiring small details, texture, or text clarity.
- Horizontal flip: Flips the image left-to-right. Disrupts questions about left/right positioning and left-to-right text reading.

## Examples ##
Question: "Is the mirror above the TV?"
Reason: The question focuses on vertical positioning. Vertical flip reverses top and bottom, making “above” mean “below,” invalidating the question. Other augmentations don't affect vertical relationships.
Choice: vertical flip

Question: "Is this photo taken indoors?"
Reason: The question requires identifying a specific environmental context. Random crop may exclude key background elements like trees, invalidating the question. Flips, color inversion, noise, and random mask don't directly affect scene context.
Choice: random crop

Question: "Are there any green beans in the image?"
Reason: The question requires identifying a specific color. Color inversion changes green to its complement, invalidating the question. Flips, noise, random mask, and random crop don't target color directly.
Choice: color inversion

Question: "How many people are in the image?"
Reason: The question requires counting visible people. Random mask can completely obscure one or more people, making the exact count impossible. Noise obscures details but typically doesn't hide entire objects, allowing approximate counting. Flips and color inversion don't affect object visibility or count.
Choice: random mask

Question: "Is the cat on the right side of the laptop?"
Reason: The question relies on horizontal positioning. Horizontal flip reverses left and right, making “right” mean “left”, invalidating the question. Other augmentations don't target horizontal positions.
Choice: horizontal flip

Question: "Does this artwork exist in the form of painting?"
Reason: The question requires identifying the texture of the artwork. Noise obscures fine details, making it hard to identify the medium. Other augmentations don't target texture details.
Choice: noise

## Your Answer ##
If multiple augmentations could disrupt the question, select the one whose effect is most direct and unambiguous. You must choose one of the given augmentations following the "Reason:" and "Choice:" format.

Question: "{}"
"""

PROMPT_ICL = """You are an expert data augmentation analyst. Your task is to select the single most semantically disruptive image augmentation that most effectively invalidates the question's premise or prevents a confident answer.

## Augmentations and Their Effects ##
- Vertical flip: Flips image top-to-bottom. Disrupts questions about “above”, “below”, “under” or reading orientation.
- Color inversion: Replaces each color with its complement. Disrupts questions relying on accurate color identification.
- Random crop: Removes random parts of the image. Disrupts questions requiring global context or peripheral objects.
- Random mask: Occludes portions of the image. Disrupts object presence, count, or attribute recognition.
- Noise: Adds visual distortion. Disrupts questions requiring small details, texture, or text clarity.
- Horizontal flip: Flips the image left-to-right. Disrupts questions about left/right positioning and left-to-right text reading.

## Examples ##
Question: "Is the mirror above the TV?"
Choice: vertical flip

Question: "Is this photo taken indoors?"
Choice: random crop

Question: "Are there any green beans in the image?"
Choice: color inversion

Question: "How many people are in the image?"
Choice: random mask

Question: "Is the cat on the right side of the laptop?"
Choice: horizontal flip

Question: "Does this artwork exist in the form of painting?"
Choice: noise

## Your Answer ##
If multiple augmentations could disrupt the question, select the one whose effect is most direct and unambiguous. You must choose one of the given augmentations following the "Choice:" format.

Question: "{}"
"""

PROMPT_REASON = """You are an expert data augmentation analyst. Your task is to select the single most semantically disruptive image augmentation that most effectively invalidates the question's premise or prevents a confident answer. Provide a clear reason explaining why the augmentation is chosen, then state your final choice.

## Augmentations and Their Effects ##
- Vertical flip: Flips image top-to-bottom. Disrupts questions about “above”, “below”, “under” or reading orientation.
- Color inversion: Replaces each color with its complement. Disrupts questions relying on accurate color identification.
- Random crop: Removes random parts of the image. Disrupts questions requiring global context or peripheral objects.
- Random mask: Occludes portions of the image. Disrupts object presence, count, or attribute recognition.
- Noise: Adds visual distortion. Disrupts questions requiring small details, texture, or text clarity.
- Horizontal flip: Flips the image left-to-right. Disrupts questions about left/right positioning and left-to-right text reading.

## Your Answer ##
If multiple augmentations could disrupt the question, select the one whose effect is most direct and unambiguous. You must choose one of the given augmentations following the "Reason:" and "Choice:" format.

Question: "{}"
"""

PROMPT_BASE = """You are an expert data augmentation analyst. Your task is to select the single most semantically disruptive image augmentation that most effectively invalidates the question's premise or prevents a confident answer.

## Augmentations and Their Effects ##
- Vertical flip: Flips image top-to-bottom. Disrupts questions about “above”, “below”, “under” or reading orientation.
- Color inversion: Replaces each color with its complement. Disrupts questions relying on accurate color identification.
- Random crop: Removes random parts of the image. Disrupts questions requiring global context or peripheral objects.
- Random mask: Occludes portions of the image. Disrupts object presence, count, or attribute recognition.
- Noise: Adds visual distortion. Disrupts questions requiring small details, texture, or text clarity.
- Horizontal flip: Flips the image left-to-right. Disrupts questions about left/right positioning and left-to-right text reading.

## Your Answer ##
If multiple augmentations could disrupt the question, select the one whose effect is most direct and unambiguous. You must choose one of the given augmentations following the "Choice:" format.

Question: "{}"
"""
