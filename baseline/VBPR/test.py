from torchvision.models.resnet import ResNet50_Weights
from torchvision import models
import torch

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 打印原始模型的最后一个全连接层
print("Original model's output layer:", model.fc)

# 假设你的任务有10个类别，修改最后一个全连接层以匹配这10个类别
num_ftrs = model.fc.in_features  # 获取最后一个全连接层的输入特征数
model.fc = torch.nn.Linear(num_ftrs, 128)  # 修改最后一个全连接层

# 打印修改后的模型的最后一个全连接层
print("Modified model", model)