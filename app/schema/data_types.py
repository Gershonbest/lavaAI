from typing import TypedDict
import torch

class AudioInfo(TypedDict):
    waveforms: torch.Tensor
    sample_rate: int