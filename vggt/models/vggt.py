# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
        ┌──────────────────────────────────┐
        │        VGGT Aggregator Backbone  │
        │  (24-layer Alternating Attention)│
        │                                  │
        │  输入：patch tokens + camera token│
        │        + register token + RoPE   │
        │  输出：output_list (24层特征)     │
        └──────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────┐
        │           4 个任务头             │
        │ camera_head   ← 使用 final_features│
        │ depth_head    ← 使用 final_features│
        │ point_head    ← 使用 final_features│
        │ track_head    ← 使用 backbone 全部层│
        └──────────────────────────────────┘
在图像经过了骨干网络之后,对于最后的结果,使用4个任务头,来输出相应的预测结果
"""


import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        # 继承父类的初始化方法
        super().__init__()

        # 初始化了VGGT的骨干模型
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        
        # 初始化了VGGT的4种heads(前3个头使用的都是aggregator最后一层输出,只有track Head 使用所有层的特征)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.
        VGGT模型的前向传播

        追踪流程:
        1. 提取特征图
        2. 迭代优化追踪
        3. 返回轨迹、可见性、置信度

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
        参数:
            images (torch.Tensor): 输入图像,形状为[S, 3, H, W]或[B, S, 3, H, W],范围在[0, 1]之间。
                B: 批量大小, S: 序列长度, 3: RGB通道数, H: 高度, W: 宽度
            query_points (torch.Tensor, optional): 用于追踪的查询点,以像素坐标表示。
                形状为[N, 2]或[B, N, 2],其中N是查询点的数量。
                默认值: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        返回:
            dict: 包含以下预测结果的字典:
                - pose_enc (torch.Tensor): 相机姿态编码,形状为[B, S, 9] (来自最后一次迭代)
                - depth (torch.Tensor): 预测的深度图,形状为[B, S, H, W, 1]
                - depth_conf (torch.Tensor): 深度预测的置信度分数,形状为[B, S, H, W]
                - world_points (torch.Tensor): 每个像素的3D世界坐标,形状为[B, S, H, W, 3]
                - world_points_conf (torch.Tensor): 世界点的置信度分数,形状为[B, S, H, W]
                - images (torch.Tensor): 原始输入图像,保留用于可视化

                如果提供了query_points,还包括:
                - track (torch.Tensor): 点的轨迹,形状为[B, S, N, 2] (来自最后一次迭代),以像素坐标表示
                - vis (torch.Tensor): 追踪点的可见性分数,形状为[B, S, N]
                - conf (torch.Tensor): 追踪点的置信度分数,形状为[B, S, N]
        """        
        # 步骤1:处理输入维度
        # If without batch dimension, add it
        # 如果没有批量维度,则添加它
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # 步骤2:骨干网络特征提取
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        # 步骤3:运行各个任务头
        with torch.cuda.amp.autocast(enabled=False):
            # 3.1 相机姿态预测
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            # 3.2 深度预测
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # 3.3 点云预测
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        # 3.4 点追踪
        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        # 步骤4 推理时保存图像
        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

