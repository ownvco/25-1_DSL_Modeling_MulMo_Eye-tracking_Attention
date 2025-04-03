import torch
import torch.nn as nn
import pandas as pd
from torchvision.models import efficientnet_b0
from argparse import ArgumentParser
from PIL import Image
import torchvision.transforms as transforms
import os
from time import time

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(base_path), 'inputdata')

def parser():
    parse = ArgumentParser()
    parse.add_argument('--file_name', action = 'store')
    return parse.parse_args()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classifier_old(file_name):
    img = Image.open(os.path.join(data_path, file_name)).convert('RGB')
    img = img_transform(img)
    img = img.unsqueeze(0)

    model = mobilenet_v3_small()

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, 1)

    weight_path = os.path.join(os.path.dirname(__file__), 'pretrained_weights', 'best_weight.pth')

    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)

    start = time()

    res = torch.sigmoid(model(img)) > 0.5

    end = time()

    time_elapsed = end - start

    return res, time_elapsed

def classifier(file_name):
    img = Image.open(os.path.join(data_path, file_name)).convert('RGB')
    img = img_transform(img)
    img = img.unsqueeze(0)

    model = efficientnet_b0()

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, 1)

    weight_path = os.path.join(os.path.dirname(__file__), 'pretrained_weights', 'best_weight.pth')

    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)

    start = time()

    res = torch.sigmoid(model(img)) > 0.5

    end = time()

    time_elapsed = end - start

    return res, time_elapsed

if __name__ == '__main__':
    parse = parser()
    file_name = parse.file_name
    res, time_elapsed = classifier(file_name)
    print(f'End of Inference | elapsed time: {time_elapsed} | class label: {res}')