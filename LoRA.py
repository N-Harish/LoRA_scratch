import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize


class LoRAParameter(nn.Module):
    def __init__(self, features_in: int, features_out: int, rank:int = 1, alpha:int = 1, device:str = 'cpu'):
        super(LoRAParameter, self).__init__()
        self.lora_a = nn.Parameter(torch.zeros(rank, features_out)).to(device)
        self.lora_b = nn.Parameter(torch.zeros(features_in, rank)).to(device)

        nn.init.normal_(self.lora_a, mean=0, std=1)

        self.scale: float = alpha / rank
        self.enabled:bool = True

    def forward(self, original_weights: torch.Tensor):
        if self.enabled:
            print(type(original_weights))
            return original_weights + torch.matmul(self.lora_b, self.lora_a).view(original_weights.shape) * self.scale
        return original_weights


def lora_parametrization(layer: nn.modules, device:str = 'cpu', rank:int = 1, alpha:int = 1):
    features_in, features_out = layer.weight.shape
    return LoRAParameter(features_in=features_in, features_out=features_out, rank=rank, alpha=alpha, device=device)


def register_lora_parameter(layer, rank:int = 1, alpha:int = 1, device: str = 'cpu'):
    parametrize.register_parametrization(
        layer, 
        "weight",
        lora_parametrization(layer, device, rank=rank, alpha=alpha) 
        )
