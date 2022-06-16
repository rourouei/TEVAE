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

root_path = '/home/njuciairs/zmy/'
test_path = root_path + 'data/Oulu/'
all_paths = []
for path in os.listdir(test_path):
    all_paths.append(test_path+path)
# print(all_paths)
# image_base_folder = root_path + 'data/Oulu/Oulu-CASIA-o'
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
triplet_num = 5

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset_oulufolder(subject):
    video_folder = []
    for dir_path in all_paths:
        for i in subject:
            if i < 10:
                pfolder = dir_path + '/P00' + str(i) + '/'
            else:
                pfolder = dir_path + '/P0' + str(i) + '/'
            for emotion in emotions:
                folder = pfolder + emotion
                video_folder.append(folder)
    return video_folder

class OuluLoader(data.Dataset):
    def __init__(self, subject, spacing, transform=None):
        super(OuluLoader, self).__init__()
        self.seed = np.random.seed(567)
        self.transform = transform
        video_folder = sorted(make_dataset_oulufolder(subject))
        self.valid_video_folder = []

        # spacing = 3, triplet_num = 5
        length = spacing * triplet_num + 1
        for video_path in video_folder:
            if os.path.isdir(video_path) == False:
                continue
            frames = sorted(os.listdir(video_path))
            if len(frames) - length > 0:
                self.valid_video_folder.append(video_path)

        self.spacing = spacing
        self.data_len = len(self.valid_video_folder)

    def __getitem__(self, index):
        anchor_index = 0
        pos_index = 1
        path = self.valid_video_folder[index]
        frames = sorted(os.listdir(path))
        anchor_frame_path = path + '/' + frames[anchor_index]
        close_frame_path = path + '/' + frames[pos_index]

        far_frame_paths = []
        for i in range(1, triplet_num + 1):
            neg_index = i * self.spacing + anchor_index + 1
            far_frame_path = path + '/' + frames[neg_index]
            far_frame_paths.append(far_frame_path)

        try:
            anchor_img = Image.open(anchor_frame_path)
            close_img = Image.open(close_frame_path)

            far_imgs = []
            for far in far_frame_paths:
                far_img = Image.open(far)
                far_imgs.append(far_img)

        except FileNotFoundError:
            print("sample missing use first")
            return self.__getitem__(0)

        imgs = [0] * (2 + triplet_num)
        imgs[0] = self.transform(anchor_img)
        imgs[1] = self.transform(close_img)

        for i in range(triplet_num):
            imgs[2+i] = self.transform(far_imgs[i])
        video_path = self.valid_video_folder[index]

        return (imgs, video_path)

    def __len__(self):
        return self.data_len
