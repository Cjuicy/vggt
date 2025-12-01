# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
1. 设计了一个简单架构,最小化3D归纳偏置,使模型能够从大量3D注释数据中学习.
    
2. 具体结构为一个大型Transformer
    
3. 具体流程
    
    - 每张图像,被切成K个patch(小块)一个patch就是一个token

    - 使用DINO特征模型,把每个patch转成特征向量(DINO:强自监督视觉特征模型)           (__build_patch_embed__函数)

        - (这让后面的Transformer更容易学到几何关系(深度、相机、点图、跟踪))
            
    - 所有图像的tokens合并起来形成一个大的token集合
        
    - 交替进行两种注意力:Frame-wise Attention ↔️ Global Attention
        
        - Frame-wise Attention 每帧做注意力
            
            **帧内注意力**:让模型理解“单张图内部”的语义和几何结构
            
        - Global Self-Attention 所有帧混起来做注意力
            
            **跨帧注意力**:让模型理解跨视角几何关系(多视图几何)
            
4. **交替注意力**(Alternating-Attention)(网络设计)
    
    1. 参考标准的transformer模型,引入交替注意力(Alternating-Attention),让transformer在帧内和全局按照不同的方式关注
        
    2. 具体细节拆分
        
        1. frame-wise attention:每一张图片$t_k$ 中分成的n个块,第i个块$t_k^I$,每个块进行自注意力
            
        2. global self-attention 是所有图片的所有token放在一块进行分析
            
        
        (**探索**:1. 不同图片的内在关联 2.每一张图片的内在关联)
        
    3. 默认使用24层 frame-wise and global attention
        
    4. **注意**⚠️:架构没有使用任何交叉注意力层,只有自注意力层
"""


"""
这部分代码核心是实现交替注意力机制(Alternating-Attention)
frame-wise attention:帧内注意力:让模型理解“单张图内部”的语义和几何结构
global self-attention:跨帧注意力:让模型理解跨视角几何关系(多视图几何)
"""


#整体的流程
"""
Images (B, S, 3, H, W)
    ↓ Normalize
    图像归一化，让输入分布符合 DINOv2 / ViT 预训练分布

Patchify + reshape  →  (B*S, 3, H, W)
    ↓
    将每帧拆成 patch，同时把 Batch 与 Frame 合并，方便送入 PatchEmbed

PatchEmbed (DINOv2 / ConvNeXt / ViT)
    ↓
    输出纯视觉 patch tokens：
    patch_tokens: (B*S, P, C)

Camera Token + Register Tokens 拼接
    ↓
    shape: (B*S, 1 + R + P, C)
    camera_token：用于预测相机位姿  
    register_tokens：用于跨帧对齐、锚点作用  
    patch_tokens：视觉内容

PositionGetter → 计算每个 patch 的 (y, x) 坐标
    ↓
    pos: (B*S, P, 2)
    对 patch 生成空间网格坐标（不作用 camera/reg tokens）

RoPE（Rotary Position Embedding）
    ↓
    用 (y, x) 坐标对 patch 的 Q/K 注入 2D 相对位置信息
    具备旋转/视角几何的敏感性

┌──────────────────────────────────────────────┐
│ Alternating Attention (24 层交替注意力)       │
│                                              │
│   for layer in 1..24:                        │
│       Frame-wise Attention（帧内）            │
│           输入 shape: (B*S, P_total, C)      │
│           只看同一帧内部 patch → 提取局部几何   │
│                                              │
│       Global Attention（帧间）                │
│           reshape → (B, S*P_total, C)        │
│           所有帧的 patches 混合注意力          │
│           学习深度线索、匹配、多视角几何       │
│                                              │
│   每个 block 输出一次特征存入 output_list     │
└──────────────────────────────────────────────┘
    ↓
输出 24 个混合时空特征层：output_list
    - 每层 shape = (B, S, P_total, C)
    - 为相机、深度、点云、tracking 等头提供不同层次的几何特征

接下来的任务头（独立 MLP）：
    ↓
相机位姿预测  ĝ  (B, S, 7)
深度预测      D̂  (B, S, H', W')
点图预测      P̂  (B, S, H', W', 3)
Tracking / Correspondence（跨帧匹配）


"""



import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.

        
    这个 聚合器 模块实现了交替注意力机制，用于多帧图像的特征聚合。(详见看论文)
    AA Transformer(Alternating Attention Transformer) 通过在帧内和全局范围内交替应用注意力机制，有效地捕捉时空特征。

    记住要设置 model.train() 以启用梯度检查点以减少内存使用。

    参数:
        img_size (int): 图像的像素大小。
        patch_size (int): PatchEmbed 的每个补丁的大小。
        embed_dim (int): 令牌嵌入的维度。
        depth (int): 块的数量。
        num_heads (int): 注意力头的数量。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比率。
        num_register_tokens (int): 注册令牌的数量。
        block_fn (nn.Module): 用于注意力的块类型（默认为 Block）。
        qkv_bias (bool): 是否在 QKV 投影中包含偏置。
        proj_bias (bool): 输出投影中是否包含偏置。
        ffn_bias (bool): MLP 层中是否包含偏置。
        patch_embed (str): Patch embed 的类型。例如 "conv" 或 "dinov2_vitl14_reg"。
        aa_order (list[str]): 交替注意力的顺序，例如 ["frame", "global"]。
        aa_block_size (int): 在切换之前每种注意力类型下分组的块数。如果不必要，设置为 1。
        qk_norm (bool): 是否应用 QK 正则化。
        rope_freq (int): 旋转位置嵌入的基频率。-1 表示禁用。
        init_values (float): 层缩放的初始值。
    """

    def __init__(
        self,
        img_size=518,                       # 输入图像的像素大小(像素越高,捕捉细节能力越大,但是计算量也越大)
        patch_size=14,                      # 输入图像的补丁大小(这里默认为14,表示每个补丁大小为14x14像素)
        embed_dim=1024,                     # token嵌入的维度(每个patch被映射到1024维的向量空间,存储小块的信息)
        depth=24,                           # Transformer的层数(对应堆叠多少个交替注意力块(24层))
        num_heads=16,                       # 注意力头数量(每个token被映射到16个头,每个头关注全文不同信息,最后交会获得全局全面的信息)
        mlp_ratio=4.0,                      #MLP 隐藏层的维度与嵌入维度（embed_dim）之间的比例 (embeding dim * mlp_ratio = hidden dim,隐藏层纬度,增强模型表达能力,捕捉数据中的高级特征,更复杂的空间结构)
        num_register_tokens=4,              # 用于多帧对齐的额外token(借助register_token来进行多帧图像的对齐以便模型更好的分析和理解)
        block_fn=Block,                     # 块函数(Block 是 Transformer 中的基本单位，包含了 多头自注意力（Multi-Head Attention） 和 MLP 层。通过残差连接保证流动性)
        qkv_bias=True,                      # 是否在键值向量中添加偏置(帮助模型学习更精细的几何变换)
        proj_bias=True,                     # 是否在投影层（输出层）加入偏置项。(帮助模型更好的调整输出)
        ffn_bias=True,                      # 是否在FFN层（Feed-Forward Network）中加入偏置项。(帮助模型更好的调整输出)    
        patch_embed="dinov2_vitl14_reg",    # 决定了对patch块的处理方式(使用DINO特征模型把每个patch转成特征向量,更好的提取图片块的语义和几何信息)(dino是当时最先进的自监督视觉特征模型)
        aa_order=["frame", "global"],       # 这里确定了交替注意力的顺序(frame-wise attention ↔️ global attention,单张图片的帧内注意力和跨多张图片的全局注意力交替进行)
        aa_block_size=1,                    # aa_block_size 控制了每种类型的注意力（frame-wise 和 global attention）切换的频率。(默认为1表示每个块交替进行)
        qk_norm=True,                       # 是否在 Q 和 K 的内积计算中应用归一化。(帮助稳定训练过程,QK有时候数值会过大导致不稳定)
        rope_freq=100,                      # 旋转编码的频率。(默认为100)(此编码是通过正弦和余弦函数将位置信息编码到token中,帮助模型理解空间关系)
        init_values=0.01,                   # 初始权重值(决定了每一层的学习速率,较小的值有助于稳定训练过程)
    ):
        # 初始化
        super().__init__()
        
        # 构建patch嵌入层(将图片转换成特征向量,提取图片中的信息)  
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        # 如果频率大于0，则初始化旋转位置嵌入(RoPE作用点:将位置信息编码到token中)
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # 初始化两种不同的注意力层
        self.frame_blocks = nn.ModuleList(      #实现帧内注意力
            [
                block_fn(                       #沿用前面设置好的参数
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)           # 初始化24个帧内注意力层
            ]
        )

        self.global_blocks = nn.ModuleList(     # 实现全局注意力层
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)           # 初始化24个全局注意力层
            ]
        )

        self.depth = depth                      # 层数
        self.aa_order = aa_order                # 交替注意力顺序
        self.patch_size = patch_size            # patch大小
        self.aa_block_size = aa_block_size      # 交替注意力块大小 (如果为2则为2层帧内+2层全局依次堆叠,如果为1则为1层帧内+1层全局依次堆叠)

        # Validate that depth is divisible by aa_block_size
        # 验证层数是否可以被交替注意力块大小整除
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size    # 交替注意力块数量(就是帧内+全局组合在一块的块的数量)

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        # 注意：我们有两个camera token 第一个是第一帧的, 第二个是剩余的帧的 (第一帧就像论文所说用作:reference frame)
        #   camera token:用于后面预测相机位姿的, register token:用于多帧对齐的
        #   同样的适用于register token
        #   register token:用于学习夸帧对齐(同样的,第一个用于第一帧的,第二个是剩余帧的)
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        # patch tokens开始于camera和register tokens之后
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        # 初始化参数(使用小值初始化参数)
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        # 将图像标准化所需的常数注册为缓冲区(缓冲区会随着CPU、GPU之间自动移动,移到相应设备上,加快训练速度)
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False  #控制PyTorch的梯度检查点(非重入模式,更高效,更安全,避免一些潜在的错误和复杂性)

    def __build_patch_embed__(
        self,
        patch_embed,                    #块转特征的方式
        img_size,                       #图片大小
        patch_size,                     #块大小
        num_register_tokens,            #多帧对齐的额外token
        interpolate_antialias=True,     # 是否使用反锯齿插值(抗锯齿可以减少图像缩放时产生的锯齿效应和混叠现象,保持图像质量)
        interpolate_offset=0.0,         # 插值偏移量(用于调整插值过程中的位置偏移,以获得更准确的结果)
        block_chunks=0,                 # 块的切分数量(用于将块的嵌入向量拆分为多个块，并分别处理)(此设置可以用于内存优化,内存受限时使用)
        init_values=1.0,                # 初始权重值
        embed_dim=1024,                 #嵌入维度(转为特征向量的维度)
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.

        构建patch嵌入层。 如果 'conv'，我们使用一个简单的 PatchEmbed conv 层。 否则，我们使用一个视觉变换器(vision transformer)。

        参数:
            patch_embed (str): Patch embed 的类型。例如 "conv" 或 "dinov2_vitl14_reg"。(选择dinov2模型作为patch embed,转向量特征)
            img_size (int): 输入图像的像素大小。
            patch_size (int): PatchEmbed 的每个块的大小。
            num_register_tokens (int): 注册令牌数量。
            interpolate_antialias (bool): 是否使用反锯齿插值。
        """

        #对模型进行patch进行特征提取(卷积方式 or DINO方式)(VGGT论文中使用DINO方式)
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {                          #需要说明的是,dinov2模型 是一个预训练模型,是能够跟着整个模型的训练过程进行微调的(是能够进行梯度)
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](             #选择对应的dinov2模型)(这里设置self.patch_embed 就是将特征提取部分网络融合在大的VGGT模型中)
                img_size=img_size,                                  #图片大小
                patch_size=patch_size,                              #块大小
                num_register_tokens=num_register_tokens,            #多帧对齐的额外token
                interpolate_antialias=interpolate_antialias,        # 是否使用反锯齿插值(抗锯齿可以减少图像缩放时产生的锯齿效应和混叠现象,保持图像质量)
                interpolate_offset=interpolate_offset,              # 插值偏移量(用于调整插值过程中的位置偏移,以获得更准确的结果)
                block_chunks=block_chunks,                          # 块的切分数量(用于将块的嵌入向量拆分为多个块，并分别处理)(此设置可以用于内存优化,内存受限时使用)
                init_values=init_values,                            # 初始权重值
            )

            # Disable gradient updates for mask token
            # DINO模型中有一个mask token,用于处理被遮挡的patch,这里冻结它的梯度(只微调特征提取部分,不更新mask token部分)
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        功能: 实现交替注意力机制
        向前传播函数，处理输入图像并应用交替注意力机制。
        1. 对输入图像进行标准化并转换为 patch tokens
        2. 添加camera 和 register 特殊tokens(用于多帧对齐)
        3. 根据rope配置添加位置编码
        4. 交替执行帧内注意力和全局注意力(frame-wise attention 和 global attention)
        5. 将中间结果拼接并返回(融合帧内注意力和全局注意力,实现交替注意机制)

        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.

        参数:
            images (torch.Tensor): 输入图像，形状为 [B, S, 3, H, W]，范围在 [0, 1] 之间。
                B: 批量大小，S: 序列长度，3: RGB 通道，H: 高度，W: 宽度
        返回:
            (list[torch.Tensor], int):
                中间结果拼接并返回(融合帧内注意力和全局注意力,实现交替注意机制)
        """
        #获取图像的信息(B: 批量大小，S: 序列长度，3: RGB 通道，H: 高度，W: 宽度)
        B, S, C_in, H, W = images.shape

        # 如果图片的通道数不是3，则抛出错误
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        # 对输入图像进行标准化,用于patch embedding
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        # 将图片转换为 patch tokens(将patch块转换为特征向量)
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        #如果patch_tokens 是一个字典，则将其转换为 patch tokens
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        #获取patch tokens的形状信息(_, P, C)(下划线表示:我们不需要使用这个维度)
        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        # 添加camera 和 register 特殊tokens(用于多帧对齐)
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        # 将camera 和 register 添加到 patch tokens中
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)


        #获取和设置获取位置编码信息(如果启用了RoPE,则生成对应的位置信息)
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            # 不要在特殊的token(camera 和 register tokens)中使用位置编码
            # 将位置编码对于这两个token上设置为0
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        # 更新P,因为添加了特殊token
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:             # 遍历帧内注意力和全局注意力
                if attn_type == "frame":                # 帧内注意力
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(     # 处理帧内注意力
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":             # 全局注意力
                    tokens, global_idx, global_intermediates = self._process_global_attention(  # 处理全局注意力
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                # 将帧内注意力和全局注意力进行拼接
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        处理帧内注意力块。 我们保持tokens的形状为(B*S, P, C)。
        B: batch size
        S: 每个batch 中的帧数(图片数量)
        P: 每个帧的patch数量(DINO patch tokens)
        C: token 编码的维度
        
        具体开始进行帧内注意力,开始预测结果同时进行梯度优化
        """
        # If needed, reshape tokens or positions:
        # 如果必要，重新调整tokens或位置
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        # 默认的情况下，self.aa_block_size=1，每次处理一个块
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)  # 训练模式下使用checkpoint技术节省内存,避免保存所有中间激活值
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        处理全局注意力块。 我们保持tokens的形状为(B, S*P, C)。

        具体进行全局注意力,开始预测结果同时进行梯度优化
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        # 默认情况下，self.aa_block_size=1，每次处理一个块
        for _ in range(self.aa_block_size):
            if self.training:           #如果在训练,则使用checkpoint技术节省内存,避免保存所有中间激活值
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:                       #如果是预测,则直接进行预测
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for 
    
    处理特殊token，形状为(1, 2, X, C)用于多帧处理：
    1）使用索引=0的第一个位置只用于第一帧
    2）使用索引=1的所有剩余帧(S-1帧)
    3）扩展两者以匹配批量大小B
    4）将两者连接起来形成(B, S, X, C)，其中每个序列有1个第一个位置标记
       followed by (S-1) second-position tokens

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    返回:
    torch.Tensor: 处理后的标记，形状为(B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    # 切片出"查询"标记 => 形状(1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    # 切片出"其他"标记 => 形状(1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    # 连接 => 形状(B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    # 最后展开 => 形状(B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
