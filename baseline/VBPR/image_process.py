import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
import os
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(image_paths):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)

    features_list = []

    for i in tqdm.trange(len(os.listdir(image_paths))):
        image_path = os.listdir(image_paths)[i]
        image_path = os.path.join(image_paths, image_path)
        print(image_path)
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(image)

        features = features.squeeze().cpu().numpy()
        features_list.append(features)

    return features_list

f = open('image_features.csv', 'w')
features = extract_features('../../data/MicroLens-50k/MicroLens-50k_covers')
for i in tqdm.trange(len(features)):
    line = [str(i + 1)] + list(map(str, features[i]))
    f.write(','.join(line) + '\n')
f.close()
print("Image features saved to image_features.csv")