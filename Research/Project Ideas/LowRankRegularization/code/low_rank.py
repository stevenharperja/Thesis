import torch

import torch.nn as nn
import numpy as np

def low_rank_approx(matrix,rank_to_keep):
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    S_reduced = np.zeros_like(S)
    S_reduced[:rank_to_keep] = S[:rank_to_keep]
    S_reduced = np.diag(S_reduced)
    approx_matrix = np.dot(U, np.dot(S_reduced, Vh))
    return approx_matrix

def convert_model_rank(model, rank_to_keep): #maybe use fraction for rank_to_keep instead of int.(with floor for rounding) The LASER paper does this. https://openreview.net/forum?id=ozX92bu8VA
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data.cpu().numpy()
            weight = low_rank_approx(weight, rank_to_keep)
            module.weight.data = torch.tensor(weight, dtype=module.weight.dtype, device=module.weight.device)
    return model