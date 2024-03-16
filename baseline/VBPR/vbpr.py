import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


def extract_features(image_paths):
    # 定义预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=True)
    model.eval()  # 设置为评估模式

    # 移除全连接层以获取特征向量
    model = nn.Sequential(*list(model.children())[:-1])

    # 如果CUDA可用，使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 存储特征的列表
    features_list = []

    # 为每张图片提取特征
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)  # 添加批次维度并转移到设备

        with torch.no_grad():
            features = model(image)

        # 将特征从4D张量转换为1D张量
        features = features.squeeze().cpu().numpy()
        features_list.append(features)

    return features_list
