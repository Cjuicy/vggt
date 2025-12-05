# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# 这个文件定义了 TrackHead(追踪头),用于视频中的点追踪任务(Point Tracking).

# 完整数据流
"""
VGGT主干输出: aggregated_tokens_list
         │
         ▼
  ┌──────────────────────┐
  │  Feature Extractor   │  (基于DPT)
  │  (feature_only=True) │
  └──────────────────────┘
         │
         ▼
  feature_maps [B, S, 128, H//2, W//2]
         │
         ▼
  ┌──────────────────────┐
  │  BaseTrackerPredictor│
  │  (迭代优化 4次)       │
  └──────────────────────┘
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
coord_preds vis_scores conf_scores
 (坐标)     (可见性)     (置信度)
"""

import torch.nn as nn
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.

    使用DPT head 处理 tokens,使用BaseTrackerPredictor 进行追踪
    追踪是迭代进行的,通过多次迭代细化
    """

    def __init__(
        self,
        dim_in,                             #输入维度(来自VGGT主干)
        patch_size=14,                      #图像patch大小(与VGGT主干一致)
        features=128,                       #特征通道数
        iters=4,                            #迭代优化次数
        predict_conf=True,                  #是否预测置信度
        stride=2,                           #步幅
        corr_levels=7,                      #相关性金字塔层数
        corr_radius=4,                      #相关性搜索半径
        hidden_size=384,                    #隐藏层大小
    ):
        """
        Initialize the TrackHead module.
        初始化 TrackHead 模块.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size

        # 核心组件1:特征提取器(Feature Extractor)
        # Feature extractor based on DPT architecture
        # 特征提取器基于 DPT 架构
        # Processes tokens into feature maps for tracking
        # 将 tokens 处理为用于追踪的特征图
        # 作用,将VGGT主干的token转为密集特征图,为追踪提供高质量视觉特征
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation       关键!只输出特征,不做预测
            down_ratio=2,  # Reduces spatial dimensions by factor of 2      降采样2倍(提高效率)
            pos_embed=False,                                                #使用位置编码
        )

        # 核心组件2:追踪器(Tracker)
        # Tracker module that predicts point trajectories
        # 追踪模块,预测点的轨迹
        # Takes feature maps and predicts coordinates and visibility
        # 接受特征图,预测坐标和可见性、置信度
        # 作用:基于特征图预测点的轨迹,输出坐标、可见性、置信度
        self.tracker = BaseTrackerPredictor(
            latent_dim=features,  # Match the output_dim of feature extractor
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters

    def forward(
            self, 
            aggregated_tokens_list,             # VGGT主干的多层tokens
            images, patch_start_idx,            # [B, S, C, H, W]
            query_points=None,                  # 要追踪的初始点(可选)
            iters=None                          # 迭代次数(可选)
        ):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
        """
        B, S, _, H, W = images.shape

        # 步骤1:特征提取
        # 将tokens转为特征图
        # Extract features from tokens
        # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)
        # feature_maps: (B, S, features, H//2, W//2)

        # 步骤2:设置迭代次数
        # Use default iterations if not specified
        if iters is None:
            iters = self.iters              #默认4次迭代

        # 步骤3:执行追踪
        # 迭代优化点的位置预测
        # Perform tracking using the extracted features
        coord_preds, vis_scores, conf_scores = self.tracker(
            query_points=query_points,      #初始化点的位置
            fmaps=feature_maps,             #特征图
            iters=iters                     #迭代次数    
        )

        return coord_preds, vis_scores, conf_scores
