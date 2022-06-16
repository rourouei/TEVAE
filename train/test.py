from __future__ import print_function
import argparse
import os
import os.path
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
# from sklearn.model_selection import KFold
torch.cuda.set_device(2)
import os.path
a = torch.zeros(1).cuda()
print(a)
#exit()
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# 如果是预训练，不需要修改 --encoder 否则改为1 并改load_path
# 如果是训练，需要加 --cuda
# 预训练命令 --cuda
#
layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
#default_content_layers = ['relu3_1', 'relu4_1', 'relu5_1']
default_content_layers = ['relu1_1', 'relu2_1', 'relu3_1']

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,
                    help='number of data loading workers, default=2', default=2)
parser.add_argument('--batch_size', type=int,
                    default=128, help='input batch size, default=8')
parser.add_argument('--image_size', type=int, default=64,
                    help='height/width length of the input images, default=64')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent vector z, default=100')
parser.add_argument('--nef', type=int, default=32,
                    help='number of output channels for the first encoder layer, default=32')
parser.add_argument('--ndf', type=int, default=32,
                    help='number of output channels for the first decoder layer, default=32')
parser.add_argument('--instance_norm', action='store_true',
                    help='use instance norm layer instead of batch norm')
parser.add_argument('--content_layers', type=str, nargs='?', default=None,
                    help='name of the layers to be used to compute the feature perceptual loss, default=[relu3_1, relu4_1, relu5_1]')
parser.add_argument('--nepoch', type=int, default=100,
                    help='number of epochs to train for, default=5')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam, default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--encoder', default='1',
                    help="path to encoder (to continue training)")
parser.add_argument('--decoder', default='',
                    help="path to decoder (to continue training)")
parser.add_argument('--outf', default='./output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--log_interval', type=int, default=10, help='number of iterations between each stdout logging, default=1')
parser.add_argument('--img_interval', type=int, default=100, help='number of iterations between each image saving, default=100')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
print("Random Seed: ", args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manual_seed)

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(args.ngpu)
nz = int(args.nz)
nef = int(args.nef)
ndf = int(args.ndf)
nc = 3
out_size = args.image_size // 16
if args.instance_norm:
    Normalize = nn.InstanceNorm2d
else:
    Normalize = nn.BatchNorm2d
if args.content_layers is None:
    content_layers = default_content_layers
else:
    content_layers = args.content_layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        #print('type of features:', type(self.features))
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

        # For percetual loss
        batch_size = input.size(0)
        all_outputs = []
        output = x
        for i in range(len(self.features)):
            output = self.features[i](output)
            if i == 5 or i == 12 or i == 19:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



#define network architecture
class _VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self, ngpu):
        super(_VGG, self).__init__()

        self.ngpu = ngpu
        features = models.vgg19(pretrained=True).features
        print(len(features))
        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            print(name)
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            if isinstance(output.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                output = nn.parallel.data_parallel(
                    module, output, range(self.ngpu))
            else:
                output = module(output)
            if name in content_layers:
                #print('!!!', name)
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs

class _Encoder(nn.Module):
    '''
    Encoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self, ngpu):
        super(_Encoder, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            Normalize(nef),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            Normalize(nef * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            Normalize(nef * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            Normalize(nef * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.mean = nn.Linear(nef * 8 * out_size * out_size, nz)
        self.logvar = nn.Linear(nef * 8 * out_size * out_size, nz)

    def sampler(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.encoder, input, range(self.ngpu))
            hidden = hidden.view(batch_size, -1)
            mean = nn.parallel.data_parallel(
                self.mean, hidden, range(self.ngpu))
            logvar = nn.parallel.data_parallel(
                self.logvar, hidden, range(self.ngpu))
        else:
            hidden = self.encoder(input)
            hidden = hidden.view(batch_size, -1)
            mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z, mean, logvar

class _Decoder(nn.Module):
    '''
    Decoder module, as described in :
    https://arxiv.org/abs/1610.00291
    '''

    def __init__(self, ngpu):
        super(_Decoder, self).__init__()
        self.ngpu = ngpu
        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * out_size * out_size),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            Normalize(ndf * 4, 1e-3),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            Normalize(ndf * 2, 1e-3),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            Normalize(ndf, 1e-3),
            nn.LeakyReLU(0.2, True),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(ndf, nc, 3, padding=1)
        )

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.decoder_dense, input, range(self.ngpu))
            hidden = hidden.view(batch_size, ndf * 8, out_size, out_size)
            output = nn.parallel.data_parallel(
                self.decoder_conv, input, range(self.ngpu))
        else:
            hidden = self.decoder_dense(input).view(
                batch_size, ndf * 8, out_size, out_size)
            output = self.decoder_conv(hidden)
        return output

#fine some tool function
def weights_init(m):
    '''
    Custom weights initialization called on encoder and decoder.
    '''
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight.data, a=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, std=0.015)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def parseSampledDataPoint(dp0_img, nc):
    dp0_img = dp0_img.float() / 255  # convert to float and rerange to [0,1]
    if nc == 1:
        dp0_img = dp0_img.unsqueeze(3)
    dp0_img = dp0_img.permute(0, 3, 1, 2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    return dp0_img

import cv2
def visualizeAsImages(img_list, output_dir,
                      n_sample=4, id_sample=None, dim=-1,
                      filename='myimage', nrow=2,
                      normalize=True):
    if id_sample is None:
        images = img_list[0:n_sample, :, :, :]
    else:
        images = img_list[id_sample, :, :, :]
    if dim >= 0:
        images = images[:, dim, :, :].unsqueeze(1)
    vutils.save_image(images,
                      '%s/%s' % (output_dir, filename + '.png'),
                      nrow=nrow, normalize=normalize, padding=2)
    # image = cv2.imread(output_dir + '/' + filename + '.png')
    # img = image[:,:,::-1]
    # cv2.imwrite(output_dir + '/' + filename + '.png', img)


dirCheckpoints = '/home/njuciairs/zmy/code/STVAE/checkpoints/stvae_rotate4'
dirImageoutput = dirCheckpoints + '/train_oulu_img'
dirTestingoutput = dirCheckpoints + '/test_oulu_img'
dirModel = dirCheckpoints + '/model'
load_encoder = '/home/njuciairs/zmy/code/STVAE/checkpoints/pretrain_final/model/encoder_epoch_' + str(args.nepoch) + '.pth'
load_decoder = '/home/njuciairs/zmy/code/STVAE/checkpoints/pretrain_final/model/decoder_epoch_' + str(args.nepoch) + '.pth'
dir_ran = dirCheckpoints + '/random_gen'
try:
    os.makedirs(dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(dirImageoutput)
except OSError:
    pass
try:
    os.makedirs(dirTestingoutput)
except OSError:
    pass
try:
    os.makedirs(dirModel)
except OSError:
    pass
try:
    os.makedirs(dir_ran)
except OSError:
    pass

# 用VGGFace初始化VGG网络
# define network instance
# descriptor = _VGG('VGG19')
# dirCheckpoint = torch.load('PrivateTest_model.t7')
# descriptor.load_state_dict(dirCheckpoint['net'])
# descriptor = _VGG('VGG19').load_state_dict(torch.load('PrivateTest_model.t7')['net'])
# exit()

# 初始化网络实例
descriptor = _VGG(ngpu)
encoder = _Encoder(ngpu)
encoder.apply(weights_init)
decoder = _Decoder(ngpu)
decoder.apply(weights_init)


# Load encoder and decoder
if args.encoder != '':
    print('LOAD MODEL!!!!!!!!!!!!!!!!!!!!!!', load_encoder)
    encoder.load_state_dict(torch.load(load_encoder, map_location='cpu'))
    decoder.load_state_dict(torch.load(load_decoder, map_location='cpu'))

if args.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    descriptor.cuda()

##########################################################################################
# some common things
from PIL import Image

mse = nn.MSELoss()


def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        fpl += mse(f, target.detach()).div(2.0)#.div(f.size(1))
    return fpl


# kld_criterion = nn.KLDivLoss(reduce=False)
# 初始化lr=0.005 训练五轮 每一轮衰减1/2
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(parameters, lr=args.lr, betas=(args.beta1, 0.999))

celebA_path = '../celebA'
# alpha是kl散度损失因子，beta是重建损失因子
alpha = 0.8
beta = 1.0
batch_size = 64
# margin = 0.3
gamma = 0.0
# pix_para = 0.0
# ranking_loss = nn.MarginRankingLoss(margin=margin)

# 在celeba上预训练STVAE
from dataloader import CelebA


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 20))
    print('change lr:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

kld_losses = []
fpl_losses = []
rank_losses = []
total_losses = []
kld_losses_test = []
fpl_losses_test = []
rank_losses_test = []
total_losses_test = []


import numpy as np

def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    data = np.array(datas)
    np.savetxt(file_name, data)
    # print("保存文件成功，处理结束")

# test_list = [1,2,3,4,5,6]
# data_write_csv(dirCheckpoints + '/pretrain/test.txt', test_list)

def write_pretrain_losse():
    data_write_csv(dirCheckpoints + '/kld_loss_train.txt', kld_losses)
    data_write_csv(dirCheckpoints + '/fpl_loss_train.txt', fpl_losses)
    data_write_csv(dirCheckpoints + '/rank_loss_train.txt', rank_losses)
    data_write_csv(dirCheckpoints + '/total_loss_train.txt', total_losses)

    data_write_csv(dirCheckpoints + '/kld_loss_test.txt', kld_losses_test)
    data_write_csv(dirCheckpoints + '/fpl_loss_test.txt', fpl_losses_test)
    data_write_csv(dirCheckpoints + '/rank_loss_test.txt', rank_losses_test)
    data_write_csv(dirCheckpoints + '/total_loss_test.txt', total_losses_test)


def transform_convert(img_tensor, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return img_tensor

def random_generate(path):
    size = 100
    # rm = Variable(torch.FloatTensor(1, size).normal_()).cuda()
    # rl = Variable(torch.FloatTensor(1, size).normal_()).cuda()
    # z = encoder.sampler(rm, rl)
    z = torch.FloatTensor(1, size).normal_().cuda()
    img = decoder(z)
    transform_convert(img)
    vutils.save_image(img, path)

def random_generates(n, epoch):
    for i in range(n):
        random_generate(dir_ran + '/' + str(epoch) + '_' + str(i) + '.png')

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

from sklearn.model_selection import  KFold

FOLD = 10
kfold = KFold(FOLD, shuffle=True, random_state=10)
train_split = []
test_split = []
subject = range(1, 81)
for i, (train_index, test_index) in enumerate(kfold.split(subject)):
    #print('Fold: ', i)
    train_subjects = [subject[i] for i in train_index]
    test_subjects = [subject[i] for i in test_index]
    train_split.append(train_subjects)
    test_split.append(test_subjects)
all_subject = list(range(1, 81))

# define a batch-wise l2 loss
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
    delta = 8e-1 * torch.ones(anc.size(0), device='cuda')
    # N x 5 x 100
    #print('!!!!', anchor.shape)
    anc = torch.unsqueeze(anc, dim=1)
    pos = torch.unsqueeze(pos, dim=1)
    neg = torch.unsqueeze(neg, dim=1)
    close_distance = criterion_l2(anc, pos).view(-1)
    far_distance = criterion_l2(anc, neg).view(-1)
    # print('close_distance:', close_distance[0], "far_distance:", far_distance[0], "sub:", close_distance[0] - far_distance[0])
    loss = torch.max(torch.zeros(anc.size(0), device='cuda'), (close_distance - far_distance + delta))
    return loss.mean()

from dataloader import Oulu_triplet

mytransform = transforms.Compose([
    # transforms.Scale((70, 70)),
    # transforms.RandomCrop(64),
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def test_one_pic(path, recon_path):
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, dim=0)
    img = parseSampledDataPoint(img, 3)
    z, _, _ = encoder(img)
    recon = decoder(z)
    vutils.save_image(recon, recon_path)

encoder.eval()
decoder.eval()

def interpolate(path1, path2, n, pathname):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    img1 = mytransform(img1).cuda()
    img2 = mytransform(img2).cuda()
    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)
    print('!!!!!!!!!!!!!!!!!!', img1.shape)
    z1, _, _ = encoder(img1)
    z2, _, _ = encoder(img2)
    print(z1[0].shape)

    z = torch.stack([z1 + (z2 - z1) * t for t in np.linspace(0, 1, n)])
    # z = z1
    interpolate_list = decoder(z)
    interpolate_list = transform_convert(interpolate_list)
    visualizeAsImages(interpolate_list.data.clone(),
                      '',
                      filename=pathname, n_sample=n, nrow=n, normalize=False)


dirpath = '/home/njuciairs/zmy/code/STVAE/checkpoints/originalVAE/visualize/inter/P035'
def play_inter(dirpath):
    print('!!!!!!!Call')
    for root, _, fnames in os.walk(dirpath):
        print('!!!!!!!Call')
        if len(fnames) == 0:
            continue
        validnames = []
        for fname in fnames:
            validnames.append(fname)
        fnames = sorted(validnames)
        path1 = root + '/' + fnames[0]
        path2 = root + '/' + fnames[-1]
        print(path2)
        l = ''
        imgPath = root
        if imgPath.find('Anger') != -1:
            l = 'Anger'
        elif imgPath.find('Disgust') != -1:
            l = 'Disgust'
        elif imgPath.find('Fear') != -1:
            l = 'Fear'
        elif imgPath.find('Happiness') != -1:
            l = 'Happiness'
        elif imgPath.find('Sadness') != -1:
            l = 'Sadness'
        elif imgPath.find('Surprise') != -1:  # surprise
            l = 'Surprie'
        pathname = root + '/' + 'inter_' + l
        print(path1)
        interpolate(path1, path2, 8, pathname)

# play_inter(dirpath)


# def get_z(dirpath):
#     for root, _, fnames in os.walk(dirpath):
#         if len(fnames) == 0:
#             continue
#         i = 0
#         for fname in fnames:
#             path = os.path.join(root, fname)
#             img = Image.open(path).convert('RGB')
#             img = mytransform(img).cuda()
#             img = torch.unsqueeze(img, dim=0)
#             z, _, _ = encoder(img)
#             arr = z.detach().cpu().numpy()
#             i += 1
#             txt_path = root + '/' + str(i) + '.txt'
#             np.savetxt(txt_path, arr)

dirpath_root = '/home/njuciairs/zmy/code/STVAE/checkpoints/stvae_rotate4/visualize/getz'
# play_inter(dirpath)
# get_z(dirpath)

train = []
test = [20, 35, 38, 75, 45, 7, 3, 44,2]
for i in range(1, 82):
    if i not in test:
        train.append(i)

print('train:', train)
print('test:', test)

def get_z():
    for sub in test_split:
        if sub < 10:
            dirpath = dirpath_root + '/P00' + str(sub)
        else:
            dirpath = dirpath_root + '/P0' + str(sub)
        for root, _, fnames in os.walk(dirpath):
            if len(fnames) == 0:
                continue
            i = 0
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                img = Image.open(path).convert('RGB')
                img = mytransform(img).cuda()
                img = torch.unsqueeze(img, dim=0)
                z, _, _ = encoder(img)
                arr = z.detach().cpu().numpy()
                i += 1
                txt_path = root + '/' + str(i) + '.txt'
                # print(txt_path)
                np.savetxt(txt_path, arr)

# get_z()

######################## 分类结果
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import  KFold

def get_embedding(pathname):
    img = Image.open(pathname).convert('RGB')
    img = mytransform(img).cuda()
    img = torch.unsqueeze(img, dim=0)
    z, _, _ = encoder(img)
    return z

def recon_save(z, pathname):
    img = decoder(z)
    img = transform_convert(img)
    visualizeAsImages(img.data.clone(),
                      '',
                      filename=pathname, n_sample=1, nrow=1, normalize=False)

######################### Facial expression transform

def expression_transform(id1, expr, id2, recon_path):
    z_id1 = get_embedding(id1)
    z_expr = get_embedding(expr)
    z_id2 = get_embedding(id2)
    z_recon = z_id2 + z_expr - z_id1
    recon_save(z_recon, recon_path)

dirTransform = '/home/njuciairs/zmy/code/STVAE/checkpoints/transform/3to2'

for i in range(1, 7):
    dirTransform1 = dirTransform + '/sample' + str(i)
    id1 = dirTransform + '/id1.jpeg'
    expr = dirTransform1 + '/expr.jpeg'
    id2 = dirTransform + '/id2.jpeg'
    recon_path = dirTransform1 + '/transform'
    expression_transform(id1, expr, id2, recon_path)


########################## 只用encoder进行FER任务
classifier = nn.Linear(100, 6)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# classifier.apply(weights_init)
# classifier.cuda()


def load(encoder, classifier):
    if args.encoder != '':
        print('LOAD MODEL!!!!!!!!!!!!!!!!!!!!!!')
        encoder.load_state_dict(torch.load(load_encoder, map_location='cpu'))
        classifier.apply(weights_init)
    if args.cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()
    return encoder, classifier

def expr_recog_encoder():
    best_accs = []
    accFilename = '/home/njuciairs/zmy/code/STVAE/checkpoints/evaluate/cnn/classify_acc.txt'
    accfile = open(accFilename, 'a')
    for k in range(10):
        acc = []
        load(encoder, classifier)
        for epoch in range(100):
            encoder.train()
            classifier.train()
            total = 0
            correct = 0
            dataset = Oulu_triplet.SingleOuluLoader2(train_split[k], mytransform)
            print('# size of the current (sub)dataset is %d' % len(dataset))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=4)
            for i, batch in enumerate(dataloader):
                img, label, path = batch[0].cuda(), batch[1].cuda(), batch[2]
                optimizer1.zero_grad()
                _, mu, _ = encoder(img)
                mu = F.dropout(mu, p=0.5)
                outputs = classifier(mu)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer1.step()
                # print('[{}/{}][{}/{}] LOSS: {:.4f}'.format(
                #             epoch, 100, i, len(dataloader), loss))

            dataset = Oulu_triplet.SingleOuluLoader2(test_split[k], mytransform)
            print('# size of the current (sub)dataset is %d' % len(dataset))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=4)
            encoder.eval()
            classifier.eval()
            total = 0
            correct = 0
            for i, batch in enumerate(dataloader):
                with torch.no_grad():
                    img, label, path = batch[0].cuda(), batch[1].cuda(), batch[2]
                    _, mu, _ = encoder(img)
                    outputs = classifier(mu)
                _, predicted = torch.max(outputs.data, 1)

                total += label.size(0)


                correct += predicted.eq(label.data).cpu().sum().item()
                # print('[{}/{}][{}/{}] LOSS: {:.4f}'.format(
                #     epoch, 100, i, len(dataloader), loss))
            print(correct / total)
            acc.append(correct / total)
        best_acc = max(acc)
        best_accs.append(best_acc)
        print(best_accs)
        print(k, file=accfile, end=':')
        print(best_acc, file=accfile)

    print(sum(best_accs) / 10, file=accfile)

expr_recog_encoder()





