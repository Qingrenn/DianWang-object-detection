from backbone import resnet50_fpn_backbone
import torch
net = resnet50_fpn_backbone()
inp = torch.rand((32, 3, 300, 300))
output = net(inp)
for k, v in output.items():
    print(k, v.shape)


