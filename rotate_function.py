import torch
from torch.nn.functional import grid_sample
import math


def rotate_crop(conv_weight, theta=0., padding='reflection'):
    device = conv_weight.device
    print(conv_weight.shape)
    n, c, h, w = conv_weight.shape
    if theta != 0:
        # 计算旋转角度 theta 对应的余弦值 cosa 和正弦值 sina
        cosa, sina = math.cos(theta), math.sin(theta)
        # 创建一个2x2的旋转矩阵 tf，用来描述旋转操作
        tf = conv_weight.new_tensor([[cosa, -sina], [sina, cosa]], dtype=torch.float)
        # 在设备上生成以 (-1, -1) 到 (1, 1) 范围内均匀分布的一维张量，用于表示图像的 x 和 y 坐标范围
        x_range = torch.linspace(-1, 1, w, device=device)
        y_range = torch.linspace(-1, 1, h, device=device)
        # 使用这些坐标范围创建一个网格，其中 y 表示纵向坐标，x 表示横向坐标
        y, x = torch.meshgrid(y_range, x_range)
        # 将 x 和 y 合并成一个网格张量 grid，并扩展成与输入图像相同的批次大小
        grid = torch.stack([x, y], -1).expand([n, -1, -1, -1])
        # 将网格张量重塑为二维形状，并应用旋转矩阵 tf，然后重新将其形状调整为与输入图像相同的四维形状
        grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
        # 使用双线性插值方法在变换后的网格上对输入图像进行采样，从而实现图像的旋转操作
        conv_weight = grid_sample(conv_weight, grid, 'bilinear', padding, align_corners=True)

    return conv_weight


# 测试
# 构造一个3x3卷积的核参数，[1,1,3,3]
conv3_weight = torch.tensor([[[[0.0, 0.2, 1.0],
                                [0.2, 1.0, 0.2],
                                [1.0, 0.2, 0.0]]]])
# 构造一个5x5卷积的核参数，[1,1,5,5]
conv5_weight = torch.tensor([[[[0.0, 0.1, 0.2, 0.8, 1.0],
                                [0.1, 0.2, 0.8, 1.0, 0.8],
                                [0.2, 0.8, 1.0, 0.8, 0.2],
                                [0.8, 1.0, 0.8, 0.1, 0.1],
                                [1.0, 0.8, 0.2, 0.1, 0.0]]]])
# 设置旋转角度90°
theta = 0.5 * math.pi
print(conv5_weight)
result = rotate_crop(conv_weight=conv5_weight, theta=theta)
print(result)
