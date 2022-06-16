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

import torch.nn as nn
def criterion_l2(input_f, target_f):
    # return a per batch l2 loss
    res = (input_f - target_f)
    res = res * res
    return res.sum(dim=2)


def criterion_l2_2(input_f, target_f):
    # return a per batch l2 loss
    res = (input_f - target_f)
    res = res * res
    return res.sum(dim=1)


def criterion_cos(input_f, target_f):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    return cos(input_f, target_f)


def criterion_cos2(input_f, target_f):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(input_f, target_f)

def tuplet_loss(anc, pos, neg):
    delta = 2e-1 * torch.ones(anc.size(0), device='cuda')
    # N x 5 x 100
    #print('!!!!', anchor.shape)
    anc = torch.unsqueeze(anc, dim=1)
    pos = torch.unsqueeze(pos, dim=1)
    neg = torch.unsqueeze(neg, dim=1)
    close_distance = criterion_l2(anc, pos).view(-1)
    far_distance = criterion_l2(anc, neg).view(-1)
    # print('close_distance:', close_distance[0], "far_distance:", far_distance[0], "sub:", close_distance[0] - far_distance[0])
    loss = torch.max(torch.zeros(anc.size(0), device='cuda'), (close_distance - far_distance + delta))
    # print(loss.shape)
    _valid = 0
    for l in loss:
        if l > 0:
            _valid += 1
    # print('valid:', _valid)
    # print(loss.sum() / _valid)
    if _valid != 0:
        return loss.sum() / _valid, _valid
    return loss.mean(), 0

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMG_JPGS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
IMG_PNGS = ['.png', '.PNG']

NUMPY_EXTENSIONS = ['.npy', '.NPY']

# rootDir = '/media/disk4/lxp/zmy'
rootDir = '/home/njuciairs/zmy'
test_path = rootDir + '/data/Oulu/'
all_paths = []
for path in os.listdir(test_path):
    all_paths.append(test_path+path)
# print(all_paths)
# image_base_folder = root_path + 'data/Oulu/Oulu-CASIA-o'
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
triplet_num = 5

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
# 写一个用于表情识别的dataloader
def make_single_oulu2(subject, k=3):
    data_dict = []
    # paths = ['/home/njuciairs/zmy/data/Oulu/Oulu-CASIA-o']
    for dirpath_root in all_paths:
        for i in subject:
            if i < 10:
                dirpath = dirpath_root + '/P00' + str(i)
                # (dirpath)
            else:
                dirpath = dirpath_root + '/P0' + str(i)
            for root, _, fnames in os.walk(dirpath):
                fnames.sort()
                for fname in fnames[-3:]:
                    if is_image_file(fname):
                        path_img = os.path.join(root, fname)
                        # print(path_img)
                        data_dict.append(path_img)
    return data_dict


class SingleOuluLoader2(data.Dataset):
    def __init__(self, subject, transform=None):
        super(SingleOuluLoader2, self).__init__()
        self.imgs = make_single_oulu2(subject)
        self.transforms = transform

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = Image.open(imgPath).convert('RGB')
        img = self.transforms(img)
        label = 0
        if imgPath.find('Anger') != -1:
            label = 0
        elif imgPath.find('Disgust') != -1:
            label = 1
        elif imgPath.find('Fear') != -1:
            label = 2
        elif imgPath.find('Happiness') != -1:
            label = 3
        elif imgPath.find('Sadness') != -1:
            label = 4
        elif imgPath.find('Surprise') != -1:  # surprise
            label = 5
        return img, label, imgPath

    def __len__(self):
        return len(self.imgs)

# 写一个single的dataloader，带标签，只取最后的多少帧
def make_single_oulu1(subject, k=3):
    data_dict = []
    paths = [rootDir + '/data/Oulu/Oulu-CASIA-o']
    for dirpath_root in paths:
        for i in subject:
            if i < 10:
                dirpath = dirpath_root + '/P00' + str(i)
                # (dirpath)
            else:
                dirpath = dirpath_root + '/P0' + str(i)
            for root, _, fnames in os.walk(dirpath):
                fnames.sort()
                for fname in fnames[-3:]:
                    if is_image_file(fname):
                        path_img = os.path.join(root, fname)
                        # print(path_img)
                        data_dict.append(path_img)
    return data_dict

class SingleOuluLoader1(data.Dataset):
    def __init__(self, subject, transform=None):
        super(SingleOuluLoader1, self).__init__()
        self.imgs = make_single_oulu1(subject)
        self.transforms = transform

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        idPath = imgPath[:-8] + '000.jpeg'
        img = Image.open(imgPath).convert('RGB')
        img = self.transforms(img)
        img1 = Image.open(idPath).convert('RGB')
        img1 = self.transforms(img1)
        label = 0
        if imgPath.find('Anger') != -1:
            label = 0
        elif imgPath.find('Disgust') != -1:
            label = 1
        elif imgPath.find('Fear') != -1:
            label = 2
        elif imgPath.find('Happiness') != -1:
            label = 3
        elif imgPath.find('Sadness') != -1:
            label = 4
        elif imgPath.find('Surprise') != -1:  # surprise
            label = 5
        return img, img1, label, imgPath

    def __len__(self):
        return len(self.imgs)

# 写一个single的dataloader，带标签，也是间隔k的采样
def make_single_oulu(subject, k=3):
    data_dict = []
    for dirpath_root in all_paths:
        for i in subject:
            if i < 10:
                dirpath = dirpath_root + '/P00' + str(i)
            else:
                dirpath = dirpath_root + '/P0' + str(i)
            for root, _, fnames in os.walk(dirpath):
                fnames.sort()
                # for fname in sorted(fnames):
                #     if is_image_file(fname):
                #         path_img = os.path.join(root, fname)
                anc = random.choice([0, 1, 2])
                _len = len(fnames)
                while anc < _len:
                    if is_image_file(fnames[anc]):
                        data_dict.append(os.path.join(root, fnames[anc]))
                        anc = anc + k
    return data_dict

class SingleOuluLoader(data.Dataset):
    def __init__(self, subject, transform=None):
        super(SingleOuluLoader, self).__init__()
        self.imgs = make_single_oulu(subject)
        self.transforms = transform

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = Image.open(imgPath).convert('RGB')
        img = self.transforms(img)
        label = 0
        if imgPath.find('Anger') != -1:
            label = 0
        elif imgPath.find('Disgust') != -1:
            label = 1
        elif imgPath.find('Fear') != -1:
            label = 2
        elif imgPath.find('Happiness') != -1:
            label = 3
        elif imgPath.find('Sadness') != -1:
            label = 4
        elif imgPath.find('Surprise') != -1:  # surprise
            label = 5
        return img, label, imgPath

    def __len__(self):
        return len(self.imgs)


# 每个sequence 6个
# 间隔采样
def make_triplet_half(sequence):
    triplets = []
    triplet = []
    triplet.append(sequence[0])
    triplet.append(sequence[len(sequence) // 2])
    triplet.append(sequence[-1])
    triplets.append(triplet)
    triplet = []
    triplet.append(sequence[0])
    triplet.append(sequence[len(sequence) // 2])
    triplet.append(sequence[-2])
    triplets.append(triplet)
    triplet = []
    triplet.append(sequence[0])
    triplet.append(sequence[len(sequence) // 2 + 1])
    triplet.append(sequence[-1])
    triplets.append(triplet)
    triplet = []
    triplet.append(sequence[0])
    triplet.append(sequence[len(sequence) // 2 + 1])
    triplet.append(sequence[-2])
    triplets.append(triplet)

    triplet = []
    triplet.append(sequence[1])
    triplet.append(sequence[len(sequence) // 2])
    triplet.append(sequence[-1])
    triplets.append(triplet)
    triplet = []
    triplet.append(sequence[1])
    triplet.append(sequence[len(sequence) // 2])
    triplet.append(sequence[-2])
    triplets.append(triplet)
    triplet = []
    triplet.append(sequence[1])
    triplet.append(sequence[len(sequence) // 2 + 1])
    triplet.append(sequence[-1])
    triplets.append(triplet)
    triplet = []
    triplet.append(sequence[1])
    triplet.append(sequence[len(sequence) // 2 + 1])
    triplet.append(sequence[-2])
    triplets.append(triplet)

    return triplets


def make_triplet_inter(sequence, k):
    triplets = []
    _len = len(sequence)
    anc = random.choice([0,1,2])
    pos = anc + k
    neg = pos + k
    while neg < _len:
        triplet = []
        triplet.append(sequence[anc])
        triplet.append(sequence[pos])
        triplet.append(sequence[neg])
        triplets.append(triplet)
        pos = neg
        neg += k
    return triplets

def make_triplet_oulu(subject, k):
    data_dict = []
    for dirpath_root in all_paths:
        for i in subject:
            if i < 10:
                dirpath = dirpath_root + '/P00' + str(i)
            else:
                dirpath = dirpath_root + '/P0' + str(i)
            for root, _ , fnames in os.walk(dirpath):
                fnames.sort()
                sequence = []
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path_img = os.path.join(root, fname)
                        sequence.append(path_img)

                if len(sequence) != 0:
                    # print(sequence)
                    triplets = make_triplet_inter(sequence, k)
                    for item in triplets:
                        data_dict.append(item)
    #print(data_dict)
    return data_dict

import tqdm

def top_k(num_list, n):
    # n %= len(num_list)
    pad = min(num_list) - 1  # 最小值填充
    topn_list = []
    for i in range(n):
        # topn_list.append(max(num_list))
        # if max(num_list) == 0:
        #     break
        max_idx = num_list.index(max(num_list))  # 找最大值索引
        topn_list.append(max_idx)
        num_list[max_idx] = pad  # 最大值填充
    return topn_list
class SelectTripletOuluLoader(data.Dataset):
    def __init__(self, subject, select, Encoder, transform=None):
        super(SelectTripletOuluLoader, self).__init__()
        data_dict = []
        for k in range(2, 3):
            data_dict += make_triplet_oulu(subject, k)
        if select > 0:
            res = []
            tri_losses = []
            for i, triplet in tqdm.tqdm(enumerate(data_dict)):
                with torch.no_grad():
                    anc_img = Image.open(triplet[0]).convert('RGB')
                    pos_img = Image.open(triplet[1]).convert('RGB')
                    neg_img = Image.open(triplet[2]).convert('RGB')
                    anc = transform(anc_img).cuda()
                    pos = transform(pos_img).cuda()
                    neg = transform(neg_img).cuda()
                    _, mu_anc, _ = Encoder(torch.unsqueeze(anc, dim=0))
                    _, mu_pos, _ = Encoder(torch.unsqueeze(pos, dim=0))
                    _, mu_neg, _ = Encoder(torch.unsqueeze(neg, dim=0))
                    tri_loss, _ = tuplet_loss(mu_anc, mu_pos, mu_neg)
                    tri_losses.append(tri_loss.item())
            max_idx = top_k(tri_losses, select)
            for idx in max_idx:
                res.append(data_dict[idx])
                # print(data_dict[idx])
            self.triplets = res
        else:
            self.triplets = data_dict
        self.transform = transform

    def __getitem__(self, index):
        triplet = self.triplets[index]
        anc_img = Image.open(triplet[0]).convert('RGB')
        pos_img = Image.open(triplet[1]).convert('RGB')
        neg_img = Image.open(triplet[2]).convert('RGB')
        anc = self.transform(anc_img)
        pos = self.transform(pos_img)
        neg = self.transform(neg_img)
        return anc, pos, neg

    def __len__(self):
        return len(self.triplets)


class TripletOuluLoader(data.Dataset):
    def __init__(self, subject, transform=None):
        super(TripletOuluLoader, self).__init__()
        data_dict = []
        for k in range(2, 3):
            data_dict += make_triplet_oulu(subject, k)
        self.triplets = data_dict
        self.transform = transform

    def __getitem__(self, index):
        triplet = self.triplets[index]
        anc_img = Image.open(triplet[0]).convert('RGB')
        pos_img = Image.open(triplet[1]).convert('RGB')
        neg_img = Image.open(triplet[2]).convert('RGB')
        anc = self.transform(anc_img)
        pos = self.transform(pos_img)
        neg = self.transform(neg_img)
        return anc, pos, neg

    def __len__(self):
        return len(self.triplets)

# 使用人工标注的三元组
tri = [[0,2,4],[0,2,5],[0,3,4],[0,3,5],[1,2,4],[1,2,5],[1,3,4],[1,3,5],[0,4,5],[0,4,6],[1,4,5],[1,4,6]]
def make_triplet_label(sequence):
    if len(sequence) != 3 and len(sequence) != 6 and len(sequence) != 7:
        print("Error triplet label!!!")
        # print(sequence)
        return []

    triplets = []
    if len(sequence) == 3:
        triplet = []
        triplet.append(sequence[0])
        triplet.append(sequence[1])
        triplet.append(sequence[2])
        triplets.append(triplet)

    if len(sequence) == 6:
        for t in tri[:8]:
            triplet = []
            triplet.append(sequence[t[0]])
            triplet.append(sequence[t[1]])
            triplet.append(sequence[t[2]])
            triplets.append(triplet)

    if len(sequence) == 7:
        for t in tri[:12]:
            triplet = []
            triplet.append(sequence[t[0]])
            triplet.append(sequence[t[1]])
            triplet.append(sequence[t[2]])
            triplets.append(triplet)

    return triplets


label_path = rootDir + '/data/Oulu-CASIA-o'
def make_triplet_oulu_label(subject):
    data_dict = []
    for i in subject:
        if i < 10:
            dirpath = label_path + '/P00' + str(i)
        else:
            dirpath = label_path + '/P0' + str(i)
        #print(dirpath)
        for root, _, fnames in os.walk(dirpath):
            # print(fnames)
            fnames.sort()
            sequence = []
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path_img = os.path.join(root, fname)
                    sequence.append(path_img)
            # print(sequence)
            for dirpath_root in all_paths:
                _sequence = []
                for path_img in sequence:
                    suffix = path_img.split('/')
                    _sequence.append(dirpath_root + '/' + suffix[-3] + '/' + suffix[-2] + '/' + suffix[-1])
                if len(_sequence) != 0:
                    # print(_sequence)
                    triplets = make_triplet_label(_sequence)
                    for item in triplets:
                        data_dict.append(item)

    #print(data_dict)
    return data_dict

# print('what')
#([1])

class LabelTripletOuluLoader(data.Dataset):
    def __init__(self, subject, transform=None):
        super(LabelTripletOuluLoader, self).__init__()
        self.triplets = make_triplet_oulu_label(subject)
        self.transform = transform

    def __getitem__(self, index):
        triplet = self.triplets[index]
        anc_img = Image.open(triplet[0]).convert('RGB')
        pos_img = Image.open(triplet[1]).convert('RGB')
        neg_img = Image.open(triplet[2]).convert('RGB')
        anc = self.transform(anc_img)
        pos = self.transform(pos_img)
        neg = self.transform(neg_img)
        return anc, pos, neg

    def __len__(self):
        return len(self.triplets)

# dataset = LabelTripletOuluLoader([1], None)
# print('# size of the current (sub)dataset is %d' % len(dataset))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
#                                                  num_workers=4)

s_path = rootDir + '/data/Oulu/Oulu-CASIA-o'
def get_subject():
    for i in range(1, 81):
        if i < 10:
            dirpath = s_path + '/P00' + str(i) + '/Anger/000.jpeg'
        else:
            dirpath = s_path + '/P0' + str(i) + '/Anger/000.jpeg'
        img = Image.open(dirpath)
        img.save('./subject/' + str(i) + '.jpeg')


# get_subject()

###############################
# 生成Oulu带label的数据

# dirpaths_root = '/home/njuciairs/zmy/data/Oulu/Oulu-CASIA-o'
# def make_dataset_oulu_label(dirpaths_root, subject):
#     #img_list = []     # list of path to the images
#     print(dirpaths_root)
#     #assert os.path.isdir(dirpath_root)
#     data = []
#     for dirpath_root in dirpaths_root:
#         for i in subject:
#             if i < 10:
#                 dirpath = dirpath_root + '/P00' + str(i)
#             else:
#                 dirpath = dirpath_root + '/P0' + str(i)
#             for root, _, fnames in sorted(os.walk(dirpath)):
#                 fnames.sort()
#                 length = len(fnames)
#                 if len(fnames) > 0:
#                     # for fname in fnames[-length//2:]:
#                     for fname in fnames[-3:]:
#                         if is_image_file(fname):
#                             path_img = os.path.join(root, fname)
#                             #print(path_img)
#                             data.append(path_img)
#     return data
#
# class SFVAEImageFolderClassAll_oulu(data.Dataset):
#     # an object that iterates over an image folder
#     def __init__(self, root, subject, transform=None, return_paths=False, rgb = True, resize = 64, loader=resize_loader):
#         super(SFVAEImageFolderClassAll_oulu, self).__init__()
#         imgs = make_dataset_oulu_label(root, subject)
#
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in: " + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#         self.root = root
#         self.length = len(imgs)
#         self.imgs = imgs
#         self.transform = transform
#         self.return_paths = return_paths
#         self.loader = loader
#         self.rgb = rgb
#         self.resize = resize
#
#     def __getitem__(self, index):
#         imgPath = self.imgs[index]
#         img = self.loader(imgPath, rgb = self.rgb, resize = self.resize)
#         label = 0
#         if imgPath.find('Anger') != -1:
#             label = 0
#         elif imgPath.find('Disgust') != -1:
#             label = 1
#         elif imgPath.find('Fear') != -1:
#             label = 2
#         elif imgPath.find('Happiness') != -1:
#             label = 3
#         elif imgPath.find('Sadness') != -1:
#             label = 4
#         else:  # surprise
#             label = 5
#         return imgPath, img, label
#
#     def __len__(self):
#         return len(self.imgs)