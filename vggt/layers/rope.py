# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


"""
这部分的代码首先要知道 rope 这个概念, rope_freq 这个参数的含义.(论文笔记中已经记录)
它的目的是:将位置信息编码到token中,帮助模型理解空间关系
而编码的方式:
    1. 创建一个正弦和余弦函数,将位置信息编码到token中
    2. 将这些编码后的信息与token进行相乘,实现旋转(所以是在patch 通过DINO编码成token之后进行的进一步融合位置编码信息)
    3. 将旋转后的信息与token进行相加,实现位置信息嵌入
    所以频率越高,编码的位置信息越细粒度.
"""

"""
这里的2D RoPE 是不可以梯度优化的(这部分是纯数学计算位置编码)
但是,是有相应的算法,也能够让RoPE算法进行优化,可能能够实现更好的位置编码
"""



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


"""
PositionGetter() # 生成空间位置
    ↓
(height, width) 例如 16×16 patch grid
    ↓
生成 256 个坐标：(y,x)
    ↓
复制 batch 次
    ↓
返回 shape = (B, N_patches, 2)

"""


class PositionGetter:           # 生成空间位置
    """Generates and caches 2D spatial positions for patches in a grid.
    生成并缓存2D网格中patch的空间位置.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.
    这个类高效地管理了2D网格中patch的空间坐标生成，通过缓存结果避免重复计算。

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.

    属性:
        position_cache: 存储不同网格维度的预计算位置张量字典.
        缓存结果以避免重复计算.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        """初始化位置生成器，并清空缓存"""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.
        生成批次中的patch的空间位置。

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        参数:
        batch_size: 批次中的样本数量.
        height: 2D网格的行数(patch数量)
        width: 2D网格的列数(patch数量)
        device: 目标设备

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.

        返回:
            一个张量，形状为(batch_size, height*width, 2)，
            包含每个位置在网格中的y,x坐标，重复了批次中的每个样本。
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.
    功能: 将2D空间位置编码应用于输入tokens

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.
    功能: 
    1. 该模块根据输入token的二维空间位置(比如图像中的patch的x,y坐标)来应用旋转位置编码
    2. 它分别处理垂直(y轴)和水平方向的(x轴)特征位置相关特征旋转

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    参数:
    - frequency: 旋转位置编码的基频率(默认100.0)
    - scaling_factor: 频率计算时的缩放因子(默认1.0) (对频率分布的二次调节,用来控制RoPE的“有效范围”和“旋转稳定性”)

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.

    属性:
    - base_frequency: 用于计算位置编码的基频率
    - scaling_factor: 用于缩放计算频率的因子
    - frequency_cache: 用于存储预计算频率分量的缓存
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        """初始化2D RoPE模块"""
        super().__init__()
        self.base_frequency = frequency             # 旋转位置编码的基频率
        self.scaling_factor = scaling_factor        # 频率计算时的缩放因子
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}   # 频率分量的缓存

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.
        计算旋转位置编码的频率分量。

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        参数:
        - dim: 特征维度(必须是偶数) (特征向量的维度,必须是偶数,因为RoPE只适用于偶数维度的输入向量,要分成2半进行旋转操作)
        - seq_len: 最大序列长度
        - device: 计算目标设备
        - dtype: 计算结果数据类型

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.

        返回:
        - 一个元组，包含两个张量，分别表示频率分量的cos和sin
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            # 计算频率带宽
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            # 生成位置相关频率
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            # 计算并缓存频率分量
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.
        功能: 通过将特征维度拆分并重新组合来执行特征旋转

        Args:
            x: Input tensor to rotate.
        参数:
        - x: 输入张量，需要进行特征旋转

        Returns:
            Rotated feature tensor.
        返回:
            旋转后的特征张量
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.
        功能: 在一个维度上应用1D旋转位置编码

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.
        参数:
        - tokens: 输入的token特征
        - positions: 位置索引
        - cos_comp: 用于旋转的cosine分量
        - sin_comp: 用于旋转的sine分量

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        # 嵌入位置与频率分量
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        # 执行旋转
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.
        功能: 对输入的token张量应用2D旋转位置编码

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.
        参数:
        - tokens: 输入张量，形状为(batch_size, n_heads, n_tokens, dim)，
                  特征维度(dim)必须可以被4整除
        - positions: 位置张量，形状为(batch_size, n_tokens, 2)，
                     包含每个token的y和x坐标
                      
        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.
        返回:
        - 输入张量，形状与输入相同，但已经应用了2D旋转位置编码

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        返回错误:
        - 如果输入维度无效或位置格式错误，则抛出AssertionError
        
        """
        # Validate inputs
        # 验证输入
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even" # 特征维度必须可以被2整除
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)" # 位置张量必须具有形状(batch_size, n_tokens, 2)

        # Compute feature dimension for each spatial direction
        # 计算每个空间方向的特征维度
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
        # 获取频率分量
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        # 分割特征以进行垂直和水平处理
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        # 分别对每个维度应用RoPE
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        # 组合处理后的特征
        return torch.cat((vertical_features, horizontal_features), dim=-1)
