import base64
from dataclasses import dataclass
import functools
from io import BytesIO
import json
import math
import os
from PIL import Image
import random
import sys
import time

import torch
import torch.nn.functional as F
import transformers
import numpy as np


@dataclass
class VcdConfig:
    # CD config
    cd_alpha: float
    cd_beta: float
    cd_tau: float
    crop_ratio: float
    mask_ratio: float
    noise_step: int
    cd_mode: str


def disable_torch_init():
    '''
    Disable the redundant torch default initialization to accelerate model creation.
    '''
    import torch
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    return


def get_last_qid(filepath: str) -> str:
    if os.path.getsize(filepath) == 0:
        return None
    last_line = ''
    with open(filepath, 'r') as f:
        for line in f:
            last_line = line
    last_line = json.loads(last_line.strip())
    last_qid = last_line['question_id']
    print(f"Resuming to {last_qid}")
    return last_qid


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def sigmoid_decayed_entropy(logits: torch.Tensor, gamma: float = 2.5) -> float:
    '''
    Calculates the scaled, normalized Shannon entropy for a batch of logits.

    Args:
        logits (torch.Tensor): A 2D tensor of shape [batch_size, dim].
        gamma (float): The scaling parameter (gamma > 0).
    '''
    prob = F.softmax(logits, dim=1)             # convert logits to probability
    n = prob.shape[1]                            # dimension of the probability
    p_safe = prob.clone()
    p_safe[prob == 0] = 1            # mask non-zero probabilities to avoid NaN
    entropy = torch.sum(prob * torch.log2(p_safe), dim=1)            # entropy
    entropy = F.sigmoid(gamma * entropy)
    return entropy.item()


def scaled_normalized_entropy(logits: torch.Tensor, tau: float = 1.0) -> float:
    '''
    Calculates the scaled, normalized Shannon entropy for a batch of logits.

    Args:
        logits (torch.Tensor): A 2D tensor of shape [batch_size, dim].
        tau (float): The scaling parameter (tau > 0).
    '''
    prob = F.softmax(logits, dim=1)             # convert logits to probability
    n = prob.shape[1]                            # dimension of the probability
    p_safe = prob.clone()
    p_safe[prob == 0] = 1            # mask non-zero probabilities to avoid NaN
    entropy = -torch.sum(prob * torch.log2(p_safe), dim=1)            # entropy
    normalized_h = entropy / math.log2(n)                           # normalize
    scaled_h = normalized_h ** (1 / tau)                       # scale with tau
    return scaled_h.item()


def setup_path():
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(utils_dir)
    models_dir = os.path.join(src_dir, 'models')
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    return


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    return

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        output = func(*args, **kwargs)
        end_time = time.perf_counter()
        output['runtime'] = end_time - start_time
        return output
    return wrapper
