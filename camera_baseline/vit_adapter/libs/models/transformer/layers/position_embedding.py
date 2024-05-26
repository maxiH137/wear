# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Various positional encodings for the transformer.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/positional_encoding.py
# ------------------------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
from ...builder import TRANSFORMERS


@TRANSFORMERS.register_module()
class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model.

    Please see `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for more details.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default: 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default: 2*pi.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default: 1e-6.
        offset (float): An offset added to embed when doing normalization.
        normalize (bool, optional): Whether to normalize the position embedding.
            Default: False.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set," "scale should be provided and in float or int type, " f"found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function for `PositionEmbeddingSine`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, t)`.

        Returns:
            torch.Tensor: Returned position embedding with
            shape `(bs, num_pos_feats * 2, t)`
        """
        assert mask is not None
        not_mask = ~mask
        embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            embed = (embed + self.offset) / (embed[:, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos  # [bs,t,c]

    def __repr__(self):
        rep_str = self.__class__.__name__ + "("
        rep_str += f"num_pos_feats={str(self.num_pos_feats)}, "
        rep_str += f"temperature={str(self.temperature)}, "
        rep_str += f"normalize={str(self.normalize)}, "
        rep_str += f"offset={str(self.offset)})"
        return rep_str


@TRANSFORMERS.register_module()
class PositionEmbeddingLearned(nn.Module):
    """
    Position embedding with learnable embedding weights.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default: 50.
        col_num_embed (int, optional): The dictionary size of column embeddings.
            Default: 50.
    """

    def __init__(self, num_pos_feats: int = 256, num_embed: int = 100):
        super().__init__()
        self.num_embed = num_embed
        self.num_pos_feats = num_pos_feats
        self.embed = nn.Embedding(num_embed, num_pos_feats)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, mask):
        """Forward function for `PositionEmbeddingLearned`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, t)`.

        Returns:
            torch.Tensor: Returned position embedding with
            shape `(bs, num_pos_feats * 2, t)`
        """
        bs, t = mask.shape
        emb = self.embed(torch.arange(t, device=mask.device))
        pos = emb.unsqueeze(0).repeat(bs, 1, 1)
        return pos


def get_sine_pos_embed(pos_tensor: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000):
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
    Returns:
        torch.Tensor: Returned position embedding  # noqa
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    pos_res = torch.cat(pos_res, dim=2)
    return pos_res
