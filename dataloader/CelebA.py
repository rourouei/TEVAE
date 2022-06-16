import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
#import scipy.io
import random
from torchvision import transforms, utils
import string
#import zx
from dataloader.preprocess import *

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMG_JPGS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
IMG_PNGS = ['.png', '.PNG']

NUMPY_EXTENSIONS = ['.npy', '.NPY']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def resize_loader(imgPath0, rgb=True, resize=64):
    with open(imgPath0, 'rb') as f0:
        with Image.open(f0) as img0:
            if rgb:
                img0 = img0.convert('RGB')
            if resize:
                img0 = img0.resize((resize, resize),Image.ANTIALIAS)
            img0 = np.array(img0)
    return img0

data_dict = []
def make_dataset_celebafolder(dirpath_root):
    #data_dict = []
    for root, _, fnames in sorted(os.walk(dirpath_root)):
        for fname in fnames:
            if is_image_file(fname):
                path_img = os.path.join(root, fname)
                data_dict.append(path_img)
    return data_dict

make_dataset_celebafolder('/home/njuciairs/zmy/data/celebA')
print(len(data_dict))
train_dict = data_dict[:180000]
test_dict = data_dict[180000:]
class CelebaLoader(data.Dataset):
    def __init__(self, transform=None, loader=resize_loader, resize=64, train=True):
        super(CelebaLoader, self).__init__()
        #imgs = make_dataset_celebafolder(data_path)
        length = len(train_dict)
        imgs = train_dict
        if train == False:
            length = len(test_dict)
            imgs = test_dict
        self.length = length
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.resize = resize
        self.transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        imgPath0 = self.imgs[index]
        #img0 = self.loader(imgPath0, rgb=True, resize=self.resize)
        img0 = Image.open(imgPath0).convert("RGB")
        img0 = self.transform(img0)
        return img0

    def __len__(self):
        return len(self.imgs)




