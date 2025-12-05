# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#此文件定义了激活函数模块,用于将网络的原始输出转换为有意义的预测值(深度、点云、姿态等)
# 使用场景:在DPTHead中的调用


import torch
import torch.nn.functional as F

#姿态参数激活
def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    Activate pose parameters with specified activation functions.
    激活姿态参数(平移、四元数、焦距)

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
        trans_act: Activation type for translation component
        quat_act: Activation type for quaternion component
        fl_act: Activation type for focal length component
    参数:
        pred_pose_enc: 编码的姿态参数 [平移, 四元数, 焦距](translation, quaternion, focal length)
        trans_act: 平移分量的激活类型
        quat_act: 四元数分量的激活类型
        fl_act: 焦距分量的激活类型

    Returns:
        Activated pose parameters tensor
    返回:
        激活后的姿态参数张量
    """

    # 步骤1: 分离不同的姿态参数
    T = pred_pose_enc[..., :3]              #平移(x,y,z)
    quat = pred_pose_enc[..., 3:7]          #四元数(x,y,z,w)
    fl = pred_pose_enc[..., 7:]  # or fov   #焦距或视场角

    # 步骤2: 应用各自的激活函数(分离激活函数)
    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # or fov

    # 步骤3: 拼接完整姿态
    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


# 头部输出激活(核心函数)
def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract 3D points and confidence values.
    处理网络输出,提取3D点和置信度值

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values
    参数:
        out:网络输出(B, C, H, W)
        activation: 3D点的激活类型
        conf_activation: 置信度值的激活类型
        
    Returns:
        Tuple of (3D points tensor, confidence tensor)
    返回:
        (3D点张量, 置信度张量)
    """
    # 步骤1:调整维度
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    # 从最后一个维度移动通道到第4个维度
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected

    # 步骤2:分离坐标和置信度
    # Split into xyz (first C-1 channels) and confidence (last channel)
    # 分离xyz(前C-1通道)和置信度(最后一个通道)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    # 步骤3:应用激活函数
    if activation == "norm_exp":                            # 默认,常用于深度/点云(作用: 保持方向,平滑的放大)
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # 计算范数
        xyz_normed = xyz / d                                # 归一化方向
        pts3d = xyz_normed * torch.expm1(d)                 # 缩放距离
    elif activation == "norm":                              # 单位向量
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":                               # 指数
        pts3d = torch.exp(xyz)
    elif activation == "relu":                              # 非负
        pts3d = F.relu(xyz)
    elif activation == "inv_log":                           # 常用于深度预测(作用: 保持符号,对大值更平滑)
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":                        # 特殊场景 XY不变,仅Z用inv_log(作用: XY线性变化,Z平滑变化)
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":                           # [0,1]
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":                            # 不变
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # 置信度激活
    if conf_activation == "expp1":                          # 范围: [1, +∞)(保证置信度>=1)
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":                        # 范围: (0, +∞)(保证置信度>0)
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":                      # 范围: [0, 1](归一化概率值)
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out

# 逆对数变换
def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)
    逆对数变换: sign(y) * (exp(|y|) - 1)
    作用:
    - 保留符号
    - 对大值更平滑
    - 适合距离/深度预测
    
    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
