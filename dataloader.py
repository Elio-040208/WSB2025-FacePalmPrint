import argparse
import os
import random
from collections import Counter
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_num", type=int, default=5, help="number of images per person")
opt = parser.parse_args()
print(opt)

def trans_form_test(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)
    return img

def trans_form_train(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        # transforms.RandomHorizontalFlip(),         # 随机水平翻转
        # transforms.RandomRotation(15),             # 随机旋转
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)
    return img

class TrainData(Dataset):
    def __init__(self, vein_root_dir, training=True):
        self.vein_root_dir = vein_root_dir
        self.person_path = os.listdir(self.vein_root_dir)

    def __getitem__(self, idx):
        person_name = self.person_path[idx // 20]  # 一个人的图片数
        vein_imgs_path = os.listdir(os.path.join(self.vein_root_dir, person_name))
        length2_imgs = len(vein_imgs_path)
        sample2_index = random.sample(range(length2_imgs), 1)
        vein_img_path = vein_imgs_path[sample2_index[0]]
        v_img_item_path = os.path.join(self.vein_root_dir, person_name, vein_img_path)
        v_img = cv2.imread(v_img_item_path)
        v_img = cv2.cvtColor(v_img, cv2.COLOR_BGR2RGB)  # 转换为RGB
        v_img = trans_form_train(v_img)
        return v_img, person_name

    def __len__(self):
        return len(self.person_path) * 20
b = []
a = []
class TestData(Dataset):
    def __init__(self, vein_root_dir, val=True):
        self.vein_root_dir = vein_root_dir
        self.person_path = os.listdir(self.vein_root_dir)

    def __getitem__(self, idx):
        person_name = self.person_path[idx // opt.img_num]
        a.append(person_name)
        bb = Counter(a)
        b = bb[person_name] - 1
        vein_imgs_path = os.listdir(os.path.join(self.vein_root_dir, person_name))
        length1_imgs = len(vein_imgs_path)
        if len(a) == len(vein_imgs_path):
            a.clear()
        vein_img_path = vein_imgs_path[b]
        v_img_item_path = os.path.join(self.vein_root_dir, person_name, vein_img_path)
        v_img = cv2.imread(v_img_item_path)
        v_img = cv2.cvtColor(v_img, cv2.COLOR_BGR2RGB)  # 转换为RGB
        v_img = trans_form_test(v_img)

        return v_img, person_name

    def __len__(self):
        return len(self.person_path) * opt.img_num