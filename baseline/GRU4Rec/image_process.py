import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
import os
import tqdm
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(image_paths, fwrite):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 128)

    model.eval()

    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)

    features_list = []

    for i in tqdm.trange(len(os.listdir(image_paths))):
        image_path = os.listdir(image_paths)[i]
        image_path = os.path.join(image_paths, image_path)
        #print(image_path)
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(image)

        features = features.squeeze().cpu().numpy()

        img_idx = re.search(r'/(\d+)\.jpg$', image_path).group(1)
        print(img_idx)
        line = [img_idx] + list(map(str, features))
        fwrite.write(','.join(line) + '\n')

    return features_list

f = open('image_features.csv', 'w')
features = extract_features('../../data/MicroLens-50k/MicroLens-50k_covers', f)
f.close()
print("Image features saved to image_features.csv")
