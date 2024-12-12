# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from aim.v1.torch.layers import (
    Attention,
    AttentionPoolingClassifier,
    PatchEmbed,
    ViTPreprocessor,
)

__all__ = [
    "TextPreprocessor",
    "ExtractEOS",
    "RMSNorm",
    "SwiGLUFFN",
    "AttentionPoolingClassifier",
    "ViTPreprocessor",
    "PatchEmbed",
    "Attention",
]

class RMSNorm(Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y = \frac{x}{\sqrt{\mathrm{RMS}[x] + \epsilon}} * \gamma

    The root mean squared norm is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the rms norm is computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: :func:`torch.finfo(x.dtype).eps`
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> rms_norm = nn.RMSNorm([2, 3])
        >>> input = torch.randn(2, 2, 3)
        >>> rms_norm(input)

    """
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Extra information about the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class TextPreprocessor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_context_length: int = 77,
        eos_token_id: int = 49407,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.zeros(max_context_length, embed_dim)
        )
        self.max_context_length = max_context_length
        self.eos_token_id = eos_token_id

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, N = input_ids.shape
        max_len = min(N, self.max_context_length)
        eos_token_mask = input_ids == self.eos_token_id
        tokens = self.text_embedding(input_ids)
        tokens = tokens[:, :max_len] + self.positional_embedding[:max_len].unsqueeze(0)
        return tokens, eos_token_mask


class ExtractEOS(nn.Module):
    def forward(
        self, tokens: torch.Tensor, eos_token_mask: torch.Tensor
    ) -> torch.Tensor:
        B, _, D = tokens.shape
        eos_token_mask = torch.argmax(eos_token_mask.float(), dim=-1)
        assert eos_token_mask.shape == (B,)
        eos_token_mask = eos_token_mask.reshape(B, 1, 1).expand(B, 1, D)
        eos_token = torch.gather(tokens, 1, eos_token_mask)
        eos_token = eos_token.squeeze(1)
        return eos_token


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        use_bias: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        **_: Any,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=use_bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=use_bias)
        self.norm_layer = norm_layer(hidden_features) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.norm_layer(x)
        x = self.fc2(x)
        return x
