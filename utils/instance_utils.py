import torch
import numpy as np
from torch.nn.functional import log_softmax, softmax

def GAIR(num_steps, Kappa, Lambda, func):
    # Weight assign
    if func == "Tanh":
        reweight = ((Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).tanh()+1)/2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Sigmoid":
        reweight = (Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Discrete":
        reweight = ((num_steps+1)-Kappa)/(num_steps+1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()
            
    return normalized_reweight


def entropy_reweight(logits, tau):
    entropy_weight = (- softmax(logits / tau, dim=-1) * log_softmax(logits / tau, dim=-1)).sum(dim=-1)
    entropy_sum = entropy_weight.sum() / len(logits)
    return entropy_weight / entropy_sum


def focal_reweight(logits):
    pass