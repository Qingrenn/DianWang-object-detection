import os
from collections import OrderedDict

import torch
from torch.functional import norm
import torch.nn as nn
from torch.jit.annotations import List, Dict
from torch.nn import parameter
from torch.nn.modules.linear import Identity

from future_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class ResBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        
        # batchnorm is learnable
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                                kernel_size=3, stride=1, bias=False)
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(x)

        out = self.conv2(x)
        out = self.bn2(x)
        out = self.relu(x)
        
        out = self.conv3(x)
        out = self.bn3(x)
        
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()

        norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer

        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=1)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, model='fanout', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x)
            x = self.fc(x)

        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extral_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extral_blocks is None:
            extral_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channel_list=in_channels_list,
            out_channels=out_channels,
            extral_blocks=extral_blocks,
        )

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

def resnet50_fpn_backbone(pretrain_path="",
                            trainable_layers=3,
                            returned_layers=None,
                            extra_block=None):
    """
    pretrain_path: Resnet50预训练权重
    norm_layer: batch较小使用BatchNorm2d
    trainable_layers: 训练Resnet中的后多少层(1～5)
    """
    resnet_backbone = ResNet(ResBlock, [3, 4, 6, 3], include_top=False)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "[ERROR] {} is not exist.".format(pretrain_path)
        # load pretraining weigths
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))
    
    assert 0 <= trainable_layers <= 5, "[ERROR] trainable_layers {} is out of range".format(trainable_layers)
    layers2train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    
    # 如果训练所有层，那么bn1也要加入
    if trainable_layers == 5:
        layers2train.append("bn1")

    # freeze layers
    for name, parameter in resnet_backbone.named_children():
        if all([not name.startswith(layer) for layer in layers2train]):
            parameter.requires_grad_(False)
    
    if extra_block is None:
        extra_block = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1,2,3,4]
    
    assert min(returned_layers) > 0 and max(returned_layers) < 5, "[ERROR] returned_layers is out of range"
    
     # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = resnet_backbone.in_channel // 8 # 2048//8=256
    in_channels_list = [in_channels_stage2*2**(i-1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_block)


if __name__ == "__main__":
    net = resnet50_fpn_backbone()
    from torchkeras import summary
    print(summary(net, (64, 300, 300, 3)))



