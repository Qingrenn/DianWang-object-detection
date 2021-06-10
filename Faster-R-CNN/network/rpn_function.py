from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s, ) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios, ) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor

        scales = torch.as_tensor(scales, dtype=dtype, device=device) # (1)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device) # (3)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1) # (3, 1) * (1, 1) = (3, 1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # 生成的anchors模板都是以（0, 0）为中心的
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2 # (3, 4)

        return base_anchors.round() # 四舍五入

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return 

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors
    
    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List(Tensor)
        """
        计算预测特征图对应原始图像上的所有anchors的坐标
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # 特征图中行单位像素对应原图中的像素数量
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # 特征图中列单位像素对应原图中的像素数量
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 特征图中上每个点对应原图中的坐标
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1) # (-1, 4)

            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4) # (-1, 1, 4) + (1, 3, 4) = (-1, 3, 4)
            # shifts_anchor 是 [同一坐标下的三种boxes(三个List)]
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        将计算得到的所有anchor信息进行缓存
        """
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        
        # 获取每个预测特征层的尺寸(height, width)
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        
        # 获取输入图像的高和宽
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类别和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # 计算特征层上的一步等于原始图像的步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] 
                    for g in grid_sizes]

        # 根据设置的sizes和aspect_ratios设置anchors模板
        self.set_cell_anchors(dtype, device)
        
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annonate(List[List[torch.Tensor]], [])
        for i, (imgae_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)



        
