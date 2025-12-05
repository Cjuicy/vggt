# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2

"""
DPT头(Depth/Head预测头)用于从Transformer特征预测
- 深度图D
- 点云P(X、Y、Z,confidence)
- 置信度图

DPT(Dense Prediction Transformer)结构包括
- multi-scale features
- 上采样
- positional embedding
- 3x3 conv头
"""

"""
    aggregated_tokens_list[4,11,17,23]
         │
         ├─ Layer 4  → norm → reshape → project → pos_embed → resize(4×) → out[0]
         ├─ Layer 11 → norm → reshape → project → pos_embed → resize(2×) → out[1]
         ├─ Layer 17 → norm → reshape → project → pos_embed → resize(2×) → out[2]
         └─ Layer 23 → norm → reshape → project → pos_embed → identity  → out[3]
                                                                    │
                                    ┌───────────────────────────────┘
                                    ▼
                         scratch_forward (FPN融合)
                                    │
              ┌─────────────────────┴─────────────────────┐
              │ layer4 → refine4                          │
              │ refine4 + layer3 → refine3                │
              │ refine3 + layer2 → refine2                │
              │ refine2 + layer1 → refine1                │
              │ → output_conv1 (256→128或256)              │
              └───────────────────┬───────────────────────┘
                                  ▼
                    custom_interpolate (上采样到目标尺寸)
                                  │
                                  ▼
                    _apply_pos_embed (位置编码)
                                  │
              ┌───────────────────┴───────────────────┐
              │ feature_only=True?                    │
              ├─ Yes → 返回特征                        │
              └─ No  → output_conv2 (128→32→output_dim)│
                       │                               │
                       ▼                               │
            activate_head (激活函数分离)                │
                       │                               │
                ┌──────┴──────┐                        │
                ▼              ▼                        │
             preds (预测值)         conf(置信度)         │
          [B,S,1,H,W]   [B,S,1,H,W]                    │
                └──────────────┘                        │
                       │                                │
                       ▼                                │
                返回 (preds, conf)                       │
                                                        │
                       或                                │
                       ▼                                │
                 返回 features [B,S,C,H,W] ◄────────────┘
    """





import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.
    DTP头 用于预测密集点云任务

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.
    这个实现了"视觉变换器用于密集预测"中的架构
    DPT头从一个视觉变换器骨干网络中获取特征，并通过多尺度特征进行融合以生成密集预测结果


    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.

    参数:
        dim_in (int): 输入维度（通道数）
        patch_size (int, optional): 块大小。默认为14
        output_dim (int, optional): 输出通道数。默认为4
        activation (str, optional): 激活函数。默认为"inv_log"
        conf_activation (str, optional): 置信度激活函数。默认为"expp1"
        features (int, optional): 特征通道数。默认为256
        out_channels (List[int], optional): 输出通道数。默认为[256, 512, 1024, 1024]
        intermediate_layer_idx (List[int], optional): 层索引列表，用于DPT。
        pos_embed (bool, optional): 是否使用位置嵌入。默认为True
        feature_only (bool, optional): 如果为True，则仅返回特征，而不返回最后的 several layers 和激活头。默认为False
        down_ratio (int, optional): 输出分辨率的缩放因子。默认为1
    """

    def __init__(
        self,
        dim_in: int,                                        # 输入维度（通道数）
        patch_size: int = 14,                               # DINO块大小。默认为14(DPTHead对这个值进行上采样,把Token变成dense像素网格) 
        output_dim: int = 4,                                # 输出通道数。默认为4(不同头任务,不同输出, DepthHead:2 PointHead:4 )
        activation: str = "inv_log",                        # 主输出激活函数(深度、点图)(inv_log 常用与距离、深度,使较大值更平滑)
        conf_activation: str = "expp1",                     # 置信度激活函数(expp1 常用与分类,保证非负) 
        features: int = 256,                                # DTP中间特征通道数。默认为256
        out_channels: List[int] = [256, 512, 1024, 1024],   # DTP中间特征通道数。默认为[256, 512, 1024, 1024](多层token 投影后的通道)
        intermediate_layer_idx: List[int] = [4, 11, 17, 23], #获取不从Transformer不从层的不同的特征信息(不同层强调的信息不同,浅层关注细节,深层关注语义)
        pos_embed: bool = True,                             # 是否使用位置嵌入。默认为True
        feature_only: bool = False,                         # 是否只返回中间特征
        down_ratio: int = 1,                                # 输出分辨率的缩放因子。默认为1
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        # 从tokens中投影成不同通道的输出 out_channels: [256, 512, 1024, 1024](
        # 创建一个投影层列表,用于将不同Transformer层的特征映射到统一的通道维度)
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        # 升采样特征图的resize层(定义了用于调整特征图尺寸的层)
        self.resize_layers = nn.ModuleList([
            # 第0层: 上采样4倍 (out_channels[0]=256)
            # 用于最浅层特征，分辨率最低，需要大幅上采样
            nn.ConvTranspose2d(
                in_channels=out_channels[0], 
                out_channels=out_channels[0], 
                kernel_size=4, stride=4, padding=0
            ),
            
            # 第1层: 上采样2倍 (out_channels[1]=512)
            # 用于第二层特征
            nn.ConvTranspose2d(
                in_channels=out_channels[1], 
                out_channels=out_channels[1], 
                kernel_size=2, stride=2, padding=0
            ),
            
            # 第2层: 上采样2倍 (out_channels[2]=1024)
            # 用于第三层特征
            nn.ConvTranspose2d(
                in_channels=out_channels[2], 
                out_channels=out_channels[2], 
                kernel_size=2, stride=2, padding=0
            ),
            
            # 第3层: 恒等变换 (out_channels[3]=1024)
            # 用于最深层特征，分辨率已经合适，不需要调整
            nn.Identity(),
            
            # 第4层: 下采样2倍
            # 可选的额外下采样层（用于特殊情况）
            nn.Conv2d(
                in_channels=out_channels[3], 
                out_channels=out_channels[3], 
                kernel_size=3, stride=2, padding=1
            ),
        ])

        # 在上面从Tranformer多层提取4个尺度特征之后,使用_make_scratch 调整不同尺度特征的通道数和尺寸(方便融合)
        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        # 在scratch上附加额外的模块(创建了4个融合模块,用于实现FPN式的多尺度特征融合)
        # FPN 融合:
        # layer4  → refine4
        # refine4 + layer3 → refine3
        # refine3 + layer2 → refine2
        # refine2 + layer1 → refine1
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        
        head_features_1 = features              # 256 (融合后的特征通道数)
        head_features_2 = 32                    # 32  (解码器中间通道数)
        
        if feature_only:
            # 模式1: 只输出特征，不做预测(用于特征提取,迁移学习)
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1,      # 输入: 256
                head_features_1,      # 输出: 256 (保持不变)
                kernel_size=3, stride=1, padding=1
            )
        else:
            # 模式2: 完整的预测头
            #            融合特征 (256通道)             (保留丰富的语义信息)
            #   → output_conv1 (3×3 conv)              
            #   → 128通道                               (过渡降为)
            #   → output_conv2:
            #        ├─ 3×3 conv → 32通道               (轻量级预测)
            #        ├─ ReLU
            #        └─ 1×1 conv → output_dim通道
            #    → 最终预测 (深度: 2通道, 点云: 4通道)      (任务特定输出)
            # 第一步: 降维 256 → 128
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1,           # 输入: 256
                head_features_1 // 2,      # 输出: 128
                kernel_size=3, stride=1, padding=1
            )
            
            # 第二步: 进一步降维并预测
            self.scratch.output_conv2 = nn.Sequential(
                # 128 → 32
                nn.Conv2d(head_features_1 // 2, head_features_2, 
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                
                # 32 → output_dim (深度头:2通道, 点云头:4通道)
                nn.Conv2d(head_features_2, output_dim, 
                          kernel_size=1, stride=1, padding=0),
            )

    def forward(
            self,
            aggregated_tokens_list: List[torch.Tensor],  # 从多层Transformer提取的tokens
            images: torch.Tensor,                         # 输入图像 [B, S, 3, H, W]
            patch_start_idx: int,                         # patch tokens起始索引
            frames_chunk_size: int = 8,                   # 每次处理的帧数
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        实现DPT头的向前传播过程,并且支持将输入帧分块处理,这对于处理长视频序列或内存受限的情况很有用.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.
        参数:
            aggregated_tokens_list (List[Tensor]): 聚合的token列表,来自不同transformer层的token张量列表.
            images (Tensor): 输入图像,形状为[B, S, 3, H, W],范围在[0, 1].
            patch_start_idx (int): 块内token的起始索引.用于将块内token与其他token(如相机或寄存器)区分开.
            frames_chunk_size (int, optional): 每个块处理的帧数.如果为None或大于S


        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        返回:
            张量或元组
            - 如果feature_only=True: 特征图,形状为[B, S, C, H, W]
            - 否则: 预测和置信度组成的元组,形状为[B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        # 模式1:一次性处理所有帧(适用场景:视频序列较短或显存充足)
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        # 模式2:分块处理帧(内存优化)
        assert frames_chunk_size > 0

        # Process frames in batches
        # 分块处理帧
        all_preds = []
        all_conf = []

        # 将S帧分成多个chunk，每个chunk处理frames_chunk_size帧
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            # 处理当前chunk
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # Concatenate results along the sequence dimension
        # 拼接所有chunk的结果
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    


    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],  # VGGT主干的多层tokens
        images: torch.Tensor,                         # [B, S, 3, H, W]
        patch_start_idx: int,                         # patch tokens起始位置
        frames_start_idx: int = None,                 # 帧分块参数
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.
        通过DPT head的前向传播实现。

        This method processes a specific chunk of frames from the sequence.
        该方法处理序列中的特定帧块。

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.
        参数:
            aggregated_tokens_list (List[Tensor]): 来自不同transformer层的token张量列表.
            images (Tensor): 输入图像,形状为[B, S, 3, H, W].
            patch_start_idx (int): 块内token的起始索引.
            frames_start_idx (int, optional): 要处理的帧的起始索引.
            frames_end_idx (int, optional): 要处理的帧的结束索引

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        返回:
            张量或元组: 特征图 或 (预测, 置信度).
        """
        # 提取和预处理帧
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # 多尺度特征提取(核心循环)
        out = []
        dpt_idx = 0

        # 循环的作用
        # 1. 从不同Transformer层提取patch tokens
        # 2. 每层经过:归一化->重塑->1x1卷积投影->位置编码->尺寸调整
        # 3. 输出四个不同尺度但分辨率对齐的特征图
        for layer_idx in self.intermediate_layer_idx:  # [4, 11, 17, 23]
            # 2.1 提取指定层的patch tokens
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            
            # 2.2 如果是分块处理，选择对应的帧
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            
            # 2.3 重塑: [B, S, num_patches, dim] → [B*S, num_patches, dim]
            x = x.reshape(B * S, -1, x.shape[-1])
            
            # 2.4 LayerNorm归一化
            x = self.norm(x)
            
            # 2.5 转换为特征图: [B*S, num_patches, dim] → [B*S, dim, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            # 2.6 通道投影 (1×1卷积)
            x = self.projects[dpt_idx](x)  # dim → out_channels[dpt_idx]
            
            # 2.7 添加位置编码
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            
            # 2.8 尺度调整 (上采样/下采样)
            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        # (特征金字塔)融合多层特征(开始融合多尺度特征)
        # 3.1 FPN式自顶向下融合
        out = self.scratch_forward(out)

        # Interpolate fused output to match target image resolution.
        # 上采样到目标分辨率
        # 3.2 插值到目标分辨率
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        # 3.3 再次添加位置编码
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        #3.4 根据模式返回结果
        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        # 3.5 预测头:128→32→output_dim + 激活
        out = self.scratch.output_conv2(out)
        # 3.6 激活函数(分离预测值和置信度)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        # 3.7 重塑输出形状: [B*S, C, H, W] → [B, S, C, H, W]
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.
        通过融合块的前向传播(融合多尺度特征)

        Args:
            features (List[Tensor]): List of feature maps from different layers.
        参数:
            features (List[Tensor]): 来自不同层的特征图列表.
            
        Returns:
            Tensor: Fused feature map.
        返回:
            Tensor: 融合后的特征图.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        #通道调整
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        #自顶向下融合
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################

# 创建特征融合模块(用于实现FPN式的多尺度特征融合)
def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


# 创建调整不同尺度特征通道数的模块(用于把不同尺度的特征图投影到相同的通道数,方便后续融合)
def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """
    Residual convolution module.
    残差卷积模块
    """


    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """
    Feature fusion block.
    特征融合块
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.
        前向传播。

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output



def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    自定义插值以避免nn.functional.interpolate中的INT_MAX问题。
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
