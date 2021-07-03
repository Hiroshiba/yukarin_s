from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_s.config import ModelConfig
from yukarin_s.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def __call__(
        self,
        phoneme_list: Tensor,
        phoneme_length: Tensor,
        padded: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batch_size = len(phoneme_list)

        output = self.predictor(
            phoneme_list=phoneme_list,
            speaker_id=speaker_id,
        )

        mask = ~padded
        if self.model_config.eliminate_pause:
            mask = torch.logical_and(mask, phoneme_list != 0)

        loss = F.l1_loss(output[mask], phoneme_length[mask], reduction="none")
        loss = loss.mean()

        # report
        values = dict(loss=loss)
        if not self.training:
            weight = mask.to(torch.float32).mean() * batch_size
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
