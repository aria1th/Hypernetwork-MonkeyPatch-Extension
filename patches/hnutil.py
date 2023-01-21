import torch

import modules.shared


def find_self(self):
    for k, v in modules.shared.hypernetworks.items():
        if v == self:
            return k
    return None


def optim_to(optim:torch.optim.Optimizer, device="cpu"):
    def inplace_move(obj: torch.Tensor, target):
        if hasattr(obj, 'data'):
            obj.data = obj.data.to(target)
        if hasattr(obj, '_grad') and obj._grad is not None:
            obj._grad.data = obj._grad.data.to(target)
    if isinstance(optim, torch.optim.Optimizer):
        for param in optim.state.values():
            if isinstance(param, torch.Tensor):
                inplace_move(param, device)
            elif isinstance(param, dict):
                for subparams in param.values():
                    inplace_move(subparams, device)
    torch.cuda.empty_cache()


def parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout):
    if layer_structure is None:
        layer_structure = [1, 2, 1]
    if not use_dropout:
        return [0] * len(layer_structure)
    dropout_values = [0]
    dropout_values.extend([0.3] * (len(layer_structure) - 3))
    if last_layer_dropout:
        dropout_values.append(0.3)
    else:
        dropout_values.append(0)
    dropout_values.append(0)
    return dropout_values


def get_closest(val):
    i, j = divmod(val,64)
    return i*64 + (j!=0) * 64