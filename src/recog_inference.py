import torch
import torch.nn as nn
import cv2
import argparse
import time
import timm
from albumentations import Resize, Compose, Normalize
from albumentations.pytorch import ToTensorV2


CFG = {
    'fold_num': 20,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b5_ns',
    'img_size': 224,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2,
    'verbose_step': 1,
    'device': 'cuda:0'
}


CFG_convnext = {
    'model_arch': 'convnext_base_384_in22ft1k',
    'img_size': 384,
    'device': 'cuda:0'
}


CFG_vit = {
    'model_arch': 'vit_large_patch16_384',
    'img_size': 384,
    'device': 'cuda:0'
}

def get_test_transforms(size):
    return Compose([
            Resize(size, size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


class PillModel(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        # n_features = self.model.head.out_features
        # self.out_layer = nn.Linear(n_features, n_class)
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        # x = self.out_layer(x)
        return x
    

class PillModelConvnext(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        # n_features = self.model.classifier.in_features
        n_features = self.model.head.fc.out_features
        self.out_layer = nn.Linear(n_features, n_class)
        # self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        x = self.out_layer(x)
        return x


class PillModelVit(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        # n_features = self.model.classifier.in_features
        n_features = self.model.head.out_features
        self.out_layer = nn.Linear(n_features, n_class)
        # self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        x = self.out_layer(x)
        return x


def load_model(model_name):
    # model_name: vit_large, convnext_base, efficientnet
    if model_name == 'vit_large_patch16_384':
        model = PillModelVit(model_name, n_class=107)
    elif model_name == 'convnext_base_384_in22ft1k':
        model = PillModelConvnext(model_name, n_class=107)
    elif model_name == 'convnext_large_384_in22ft1k':
        model = PillModelConvnext(model_name, n_class=107)
    elif model_name == 'convnext_xlarge_384_in22ft1k':
        model = PillModelConvnext(model_name, n_class=107)
    elif model_name == 'tf_efficientnet_b5_ns':
        model = PillModel(model_name, n_class=107)
    elif model_name == 'tf_efficientnet_b7_ns':
        model = PillModel(model_name, n_class=107)
    return model


class Stage2:
    def __init__(self, model_name, model_path, device):
        self.device = device
        # model = PillModel(CFG['model_arch'], n_class=107)
        # model = PillModelConvnext(CFG_convnext['model_arch'], n_class=107)
        model_name = model_path.split('/')[1]
        self.model_name = model_name
        model = load_model(model_name)
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = model
        self.model.eval()

    def recog_pill_pipeline(self, img):
        img = img[:, :, ::-1]
        if self.model_name == 'efficientnet':
            size = 224
        else:
            size = 384
        img = get_test_transforms(size)(image=img)['image']
        img = img.unsqueeze(0)
        img = img.to(self.device)
        preds = self.model(img)
        # preds = torch.softmax(preds, -1)
        # preds = torch.argmax(preds)
        # preds = preds.item()
        return preds
