import sys
sys.path.append("../")

import torch
from network import AnchorsGenerator
from backbone import resnet50_fpn_backbone
from network import ImageList

anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
ag = AnchorsGenerator(anchor_sizes, aspect_ratios)
backbone = resnet50_fpn_backbone()

images = torch.rand(32, 3, 300, 300)
features = backbone(images)
features = list(features.values())
for f in features:
    print(f.shape)
print("batch: {}\nimages shape: {}".format(len(images), images[0].shape))

sizes = [img.shape[-2:] for img in images]

image_list = ImageList(images, sizes)
anchors = ag(image_list, features)
print(len(anchors), anchors[0].shape)