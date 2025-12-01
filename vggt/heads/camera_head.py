# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
camera_tokens
   ↓
小型 Transformer（trunk） ← 注释说的那个 trunk 就是这里
   ↓
线性层预测 (R, t, intrinsics)
'''

"""
CameraHead 核心逻辑

token → 标准化 → MLP → 输出相机姿态 / refine 量
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.
    相机头预测摄像机参数，使用迭代 refinment。(CameraHead是一个小型的Transformer,通过迭代预测 -> 修正 ->预测。 不断迭代,精炼)
    
    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    它使用一个序列的 transformer 块（“干”(小号主干)）对专用的摄像机token进行应用。(trunk 就是多层Transformer block 的集合)
    """


    def __init__(
        self,
        dim_in: int = 2048,                             #输入维度 2048
        trunk_depth: int = 4,                           #相机头主干深度
        pose_encoding_type: str = "absT_quaR_FoV",      #相机头参数编码类型
        num_heads: int = 16,                            #多头注意力的头数
        mlp_ratio: int = 4,                             #MLP 缩放因子(中间隐藏层相对于输入维度的扩展比例)
        init_values: float = 0.01,                      #初始化值(用来初始化LayerScale层的参数值)
        trans_act: str = "linear",                      # 控制平移分量的激活函数
        quat_act: str = "linear",                       # 控制四元数分量的激活函数
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive. # 控制焦距 (focal length) 和 视场角(Field of View)的激活函数
    ):
        super().__init__()

        # 如果相机参数编码类型是 absT_quaR_FoV,设置目标维度为 9 ,否则提示错误: 不支持这个相机参数编码类型
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        #设置对应的参数(平移,四元数,焦距,主干深度)
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        # 构建一个由 transformer 块组成的 trunk 骨干网络
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        # 对相机token 和 trunk 输出进行归一化(调整神经网络中的数据分布,稳定训练过程,减少内部协变量偏移)
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        # 可学习的空相机姿态token
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))   #创建一个全0张量,作为相机参数的初始值,后面学习模型的相机姿态参数.
        self.embed_pose = nn.Linear(self.target_dim, dim_in)                        #创建一个可学习的线性映射,将相机参数映射到神经网络的输入维度

        # Module for producing modulation parameters: shift, scale, and a gate.
        # 定义一个用于生成调制参数的模块(1. 接收姿态嵌入的特征向量 2. 通过SiLU激活函数进行非线性变换 3. 通过线性层将其扩展为3倍的维度 4. 输出被分成3个部分,分别作用偏移、缩放、门控)
        # 偏移,缩放,门控是用于自适应层归一化的调制参数
        # 偏移:类似于 LayerNorm 中的 bias 参数, 对归一化后的特征进行平移变换,允许模型根据输入内容调整特征幅度
        # 缩放:类似于 LayerNorm 中的 weight 参数,用于归一化后特征进行缩放变换,允许模型根据输入内容调整特征的幅度
        # 门控:控制调制的强度,决定是否以多少程度上应用调制,通过与调制结果相乘实现控制,使模型能够动态选择何时应用自适应归一化.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        # 用于从相机token中提取特征,不带仿射参数的自适应层归一化
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)      #对输入token做干净标准化,为AdaLN作准备
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)   #从规范化的token中预测姿态分量(或refine的增量)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.
        向前传递以预测相机参数。

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
        参数:
            aggregated_tokens_list (list): 来自网络的token张量列表;最后一个张量用于预测。
            num_iterations (int, optional): 迭代细化步骤的数量。默认为4。(控制迭代次数)

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        返回:
            list: 来自每次迭代的预测相机编码（激活后）的列表。
        """
        # Use tokens from the last block for camera prediction.
        # 使用最后一个块的tokens进行相机预测(前面的层都不要了)(是VGGT的最后一层)
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        # 提取相机tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        #将 VGGT的相机 token 传递给 trunk_fn 方法进行迭代细化预测(专门用于处理相机 token 的迭代细化预测)
        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.
        迭代细化相机姿态预测。

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.

        参数:
            pose_tokens (torch.Tensor): 归一化的相机tokens，形状为[B, S, C]。
            num_iterations (int): 细化迭代次数
            
        Returns:
            list: List of activated camera encodings from each iteration.
        返回:
            list: 来自每次迭代的激活相机编码列表。
        """

        # 获取姿态token张量的形状信息(Batch Size, Sequence Length, Channel)
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            # 在第一次迭代中使用一个学习到的空姿态。
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                # 分离之前的预测以避免通过时间反向传播。(预测的结果作为新的输入,但是不进行梯度传播)
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            # 生成调制参数并将其拆分为shift、scale和gate组件。
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            # 自适应层归一化和调制。
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            # 计算姿态编码的增量更新。
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            # 对平移、四元数和视场角应用最终激活函数。
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    调制输入张量，使用缩放和平移参数。

    作用:
    - 特征调制: 通过缩放和平移调整输入特征的分布,使模型能够动态适应不同的输入数据。
    - 自适应调整: 允许模型根据输入内容调整特征的幅度和位置.
    - 条件话处理:实现条件批归一化

    做什么: 根据scale 和 shift,对特征做“动态缩放+动态平移”,相当于用一个“条件输入”来控制token特征.
    优点: 根据 scale + shift，把 token 重新缩放和偏移，使后续网络能更好地产生正确的几何预测。

    输入 token 特征 x ─────────────┐
                                │
     scale（来自另一分支） ──► × (1+scale) ──► + shift ──► 输出
                                │
     shift（来自另一分支） ─────┘

    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
