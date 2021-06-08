from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict
"""
in_channel_list: list[int],送入FPN的特征图的通道数,[256, 512, 1024, 2048]
"""
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channel_list, out_channels, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channel_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        for m in self.children():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1) # 初始化卷积层参数
                nn.init.constant_(m.bias, 0) # 偏置设置为0

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        self.inner_blocks[idx](x)
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
                break
        return out
    
    def get_result_from_layer_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        self.layer_blocks[idx](x)
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
                break
        return out

    def forward(self, x):
        # type: (Dict[str, Tensor]) -> Dict[str, Tensor]
        """
        x: OrderedDicr[layer_name: feature_map]
        """
        names = list(x.keys())
        x = list(x.values())

        results = [] # order: highest -> lowest resolution 

        # first append the lowest resolution feature
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        # upsample
        for idx in range(len(x)-2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
        
        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        
        return out

class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """
    def forward(self, x, y, names):
        # type: (List[Tensor], List[Tensor], List[str]) -> Tuple[List[Tensor], List[str]]
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names
