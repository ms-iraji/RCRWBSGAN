
import re
import argparse
import os
import shutil
import time
import math
import logging
import torchvision.utils as vutils

import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import losses2

from utils import *
import losses
import ramps

NO_LABEL=-1


from utilsu import *





import os
import random
import numpy as np
import torch
def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
set_all_seeds(10)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
def save_images(images, size, image_path):
    return imsave(images, size, image_path)



def imsave(images, size, path): # # #
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, (255.0 * image).astype('uint8'))
def generated_weight(epoch):
    alpha = 0.0
    T1 = 10
    T2 = 60
    af = 0.3
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha

def nll_loss_neg(y_pred, y_true,CLASS_WEIGHT):
    out = torch.sum((y_true * y_pred)*CLASS_WEIGHT, dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))

def nll_loss_neg2(y_pred, y_true):
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log(( out) + 1e-6))



transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                transforms.RandomAffine(0, (1/8,0))]) # max horizontal shift by 4


mixup=0.48

def mixup_batch(mixup,real1,real2,fake):
                def one_batch():


                    data = torch.cat((real1,real2, fake))
                    ones1 = Variable(torch.ones(real1.size(0), 1))
                    ones2 = Variable(torch.ones(real2.size(0), 1))                    
                    zeros = Variable(torch.zeros(fake.size(0), 1))

                    perm = torch.randperm(data.size(0)).view(-1).long()
                    if True:
                        ones1 = ones1.cuda()
                        ones2 = ones2.cuda()
                        zeros = zeros.cuda()
                        perm = perm.cuda()
                    labels = torch.cat((ones1,ones2, zeros))
                    return data[perm], labels[perm]

                d1, l1 = one_batch()
                if mixup == 0:
                    return d1, l1
                d2, l2 = one_batch()
                alpha = Variable(torch.tensor(np.random.beta(mixup, mixup)) )

                #print(alpha)

                if True:
                    alpha = alpha.cuda()
                d = alpha * d1 + (1. - alpha) * d2
                l = alpha * l1 + (1. - alpha) * l2
                return d, l

def mixup_batch2(mixup,f,y):
                def one_batch():


                    data = f
                    perm = torch.randperm(data.size(0)).view(-1).long()
                    if True:
                        perm = perm.cuda()
                    labels =y 
                    return data[perm], labels[perm]

                d1, l1 = one_batch()
                if mixup == 0:
                    return d1, l1
                d2, l2 = one_batch()
                alpha = Variable(torch.tensor(np.random.beta(mixup, mixup)) )

                #print(alpha)

                if True:
                    alpha = alpha.cuda()
                d = alpha * d1 + (1. - alpha) * d2
                l = alpha * l1 + (1. - alpha) * l2
                return d, l


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
# batch_size * input_dim => batch_size * output_dim * input_size * input_size
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x



# batch_size * input_dim * input_size * input_size => batch_size * output_dim
class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 500),
        
        )
        initialize_weights(self)
        self.ln=nn.Linear(500, self.output_dim)
        self.sig=nn.Sigmoid()
        self.features=None

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        self.features=x.view(-1, x.size(1))
        output=self.ln(self.features)
        output=self.sig(output)
        return output












import re
import argparse
import os
import shutil
import time
import math
from itertools import repeat, cycle
import matplotlib as mpl
mpl.use('Agg')

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from collections import OrderedDict
import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle


#from utils import *
#from networks.wide_resnet import *
#from networks.wide_resnet13 import *
#from networks.lenet import *
#from networks.lenet14 import *

#from cmn import *



#import relation.losses2


parser = argparse.ArgumentParser(description='Interpolation consistency training')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=['cifar10','svhn','mnist','cinic','tiny','stl10','covid2','covid3','brain4','skin'],
                        help='dataset: cifar10 or svhn' )
parser.add_argument('--num_labeled', default=1000, type=int, metavar='L',
                    help='number of labeled samples per class')
parser.add_argument('--num_valid_samples', default=1000, type=int, metavar='V',
                    help='number of validation samples per class')
parser.add_argument('--arch', default='cnn13', type=str, help='either of cnn13, WRN28_2 , cifar_shakeshake26')
parser.add_argument('--dropout', default=0.0, type=float,
                    metavar='DO', help='dropout rate')

parser.add_argument('--sl', action='store_true',
                    help='only supervised learning: no use of unlabeled data')
parser.add_argument('--pseudo_label', choices=['single','mean_teacher'],
                        help='pseudo label generated from either a single model or mean teacher model')
parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use. can be either adam of sgd')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='max learning rate')
parser.add_argument('--initial_lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr_rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr_rampdown_epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate \
                    reaches to zero')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
parser.add_argument('--mixup_consistency', default=1.0, type=float,
                    help='consistency coeff for mixup usup loss')
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency_rampup_starts', default=30, type=int, metavar='EPOCHS',
                    help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=30, type=int, metavar='EPOCHS',
                    help='lepoch at which consistency loss ramp-up ends')
parser.add_argument('--mixup_sup_alpha', default=0.0, type=float,
                    help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
parser.add_argument('--mixup_usup_alpha', default=0.0, type=float,
                    help='for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
parser.add_argument('--mixup_hidden', action='store_true',
                    help='apply mixup in hidden layers')
parser.add_argument('--num_mix_layer', default=3, type=int,
                    help='number of layers on which mixup is applied including input layer')
parser.add_argument('--checkpoint_epochs', default=50, type=int,
                    metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
parser.add_argument('--evaluation_epochs', default=1, type=int,
                    metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate model on evaluation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--data_dir', type = str, default = 'data/cifar10/',
                        help='folder where data is stored')
parser.add_argument('--n_cpus', default=0, type=int,
                    help='number of cpus for data loading')
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--add_name', type=str, default='')


args = parser.parse_args()
print (args)
use_cuda = torch.cuda.is_available()


best_prec1 = 0
global_step = 0
best_acc=0
best_accun=0
##get number of updates etc#####
   
if args.dataset == 'mnist':
        z_dim=62
        G = generator(input_dim=z_dim, output_dim=1, input_size=28)
        D = discriminator(input_dim=1, output_dim=1, input_size=28)   
             
else:    

        z_dim=100

        G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        D = discriminator(input_dim=3, output_dim=1, input_size=32)
if args.dataset == 'tiny':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=3, input_size=64)
        D = discriminator(input_dim=3, output_dim=1, input_size=64)
if args.dataset == 'stl10':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=3, input_size=96)
        D = discriminator(input_dim=3, output_dim=1, input_size=96)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32)            

if args.dataset == 'covid2':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=1, input_size=64)
        D = discriminator(input_dim=1, output_dim=1, input_size=64)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32) 

if args.dataset == 'covid3':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=1, input_size=32)
        D = discriminator(input_dim=1, output_dim=1, input_size=32)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32) 

if args.dataset == 'brain4':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=1, input_size=32)
        D = discriminator(input_dim=1, output_dim=1, input_size=32)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32)    

if args.dataset == 'skin':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=3, input_size=96)
        D = discriminator(input_dim=3, output_dim=1, input_size=96)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32)                
if args.resume=='y':
        D.load_state_dict(torch.load('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/experiments/s/_D.pkl'))
        G.load_state_dict(torch.load('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/experiments/s/_G.pkl'))

#else:    
        from dcgan import Discriminator, Generator

        #D   = Discriminator(1)
        #G = Generator(1)
        #z_dim=100
G.cuda()
D.cuda()

lrG=0.0002
lrD=0.0002
beta1=0.5
beta2=0.999
G_optimizer = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lrD, betas=(beta1, beta2))

BCEloss = nn.BCELoss().cuda()
l2loss = nn.MSELoss()



D.train()
G.train()
if args.dataset == 'cifar10':
    len_data = args.num_labeled
    from dataloadercfar10 import dataloader
    num_updates = int((50000/args.batch_size))*args.epochs 
elif args.dataset == 'svhn':
    from dataloadersvh import dataloader
    len_data = args.num_labeled
    num_updates = int((73250/args.batch_size)+1)*args.epochs
elif args.dataset == 'mnist':
    from dataloadersvh import dataloader
    len_data = args.num_labeled
    num_updates = int((50000/args.batch_size)+1)*args.epochs     
    print ('number of updates', num_updates)
elif args.dataset == 'cinic':
    len_data = args.num_labeled
    num_updates = int((180000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)
elif args.dataset == 'covid2':
    len_data = args.num_labeled
    num_updates = int((5000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates) 
elif args.dataset == 'covid3':
    len_data = args.num_labeled
    num_updates = int((1000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)     
elif args.dataset == 'brain4':
    len_data = args.num_labeled
    num_updates = int((2700/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)       
elif args.dataset == 'skin':
    len_data = args.num_labeled
    num_updates = int((29812/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)               
elif args.dataset == 'tiny':
    len_data = args.num_labeled
    num_updates = int((100000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)    
elif args.dataset == 'stl10':
    len_data = args.num_labeled
    num_updates = int((105000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)     
#print (args.batch_size, num_updates, args.epochs)

#### load data###
if args.dataset == 'cifar10':
    data_source_dir = args.data_dir
    #trainloader, unlabelledloader, testloader = dataloader('cifar10', 32, args.batch_size, args.num_labeled)
    #num_classes=10
    #validloader=testloader
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')
if args.dataset == 'cinic':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cinic', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=10
    #validloader=testloader
    #trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')
if args.dataset == 'covid2':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'covid2', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=2
if args.dataset == 'covid3':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'covid3', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=3    
    #validloader=testloader
    #trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')    
if args.dataset == 'brain4':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'brain4', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=4        
if args.dataset == 'skin':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'skin', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    trainloadera, validloadera, unlabelledloadera, testloaderua, num_classes = load_data_subseta(1, args.batch_size, args.n_cpus ,'skin', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=6          
if args.dataset == 'tiny':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'tiny', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=10
    #validloader=testloader
    #trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')   
if args.dataset == 'stl10':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'stl10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=10         
if args.dataset == 'svhn':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'svhn', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #trainloader, unlabelledloader, testloader = dataloader('svhn', 32, args.batch_size, args.num_labeled)
    #num_classes=10
    #validloader=testloader
if args.dataset == 'mnist':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'mnist', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
### lists for collecting output statistics###
### lists for collecting output statistics###
wcw2 = np.ones(num_classes)
CLASS_NUM = [657, 760, 46, 219, 172, 211]
normedWeights = [1 - (x / sum(CLASS_NUM)) for x in CLASS_NUM]
print(normedWeights)
normedWeights
CLASS_WEIGHT= torch.Tensor(normedWeights).cuda()
print(wcw2)
wcw2 = torch.Tensor(wcw2 ).cuda()

train_class_loss_list = []
train_ema_class_loss_list = []
train_mixup_consistency_loss_list = []
train_mixup_consistency_coeff_list = []
train_error_list = []
train_ema_error_list = []
train_lr_list = []


val_class_loss_list = []
val_error_list = []
val_ema_class_loss_list = []
val_ema_error_list = []


### get net####

# torch.randn(20, 3, 64, 64)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 512 * block.expansion
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out#, emb
    def get_embedding_dim(self):
        return self.embDim

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

from models import DenseNet121
from lenet14 import *
from stl import *
def getNetwork(args, num_classes, ema= False):
    
    if args.arch in ['cnn13','WRN28_2']:
        net = eval(args.arch)(num_classes, args.dropout)
    elif args.arch in ['cmn']:
        net = cmn() 
    elif args.arch in ['res']:
        #net = ResNet18() 
        net=stl10(n_channel=32)  
    elif args.arch in ['dense']:
        #net = ResNet18()
        net=DenseNet121(out_size=6, mode='U-Ones', drop_rate=0.2)                       
    elif args.arch in ['cifar_shakeshake26']:
        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        net = model_factory(**model_params)
    else:
        print('Error : Network should be either [cnn13/ WRN28_2 / cifar_shakeshake26')
        sys.exit(0)
    if args.resume=='y':
        net.load_state_dict(torch.load('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/experiments/s/_C.pkl'))
    if ema:
        for param in net.parameters():
            param.detach_()

    return net



def experiment_name(sl = False,
                    dataset='cifar10',
                    labels  = 1000,
                    valid = 1000,
                    optimizer = 'sgd',
                    lr = 0.0001,
                    init_lr = 0.0,
                    lr_rampup = 5,
                    lr_rampdown = 10,
                    l2 = 0.0005,
                    ema_decay = 0.999,
                    mixup_consistency = 1.0,
                    consistency_type = 'mse',
                    consistency_rampup_s = 30,
                    consistency_rampup_e = 30,
                    mixup_sup_alpha = 1.0,
                    mixup_usup_alpha = 2.0,
                    mixup_hidden = False,
                    num_mix_layer = 3, 
                    pseudo_label = 'single',
                    epochs=10,
                    batch_size =100,
                    arch = 'WRN28_2',
                    dropout = 0.5, 
                    nesterov = True,
                    job_id=None,
                    add_name=''):
    if sl:
        exp_name = 'SL_'
    else:
        exp_name = 'SSL_'
    exp_name += str(dataset)
    exp_name += '_labels_' + str(labels)
    exp_name += '_valids_' + str(valid)
    
    exp_name += '_arch'+ str(arch)
    exp_name += '_do'+ str(dropout)
    exp_name += '_opt'+ str(optimizer)
    exp_name += '_lr_'+str(lr)
    exp_name += '_init_lr_'+ str(init_lr)
    exp_name += '_ramp_up_'+ str(lr_rampup)
    exp_name += '_ramp_dn_'+ str(lr_rampdown)
    
    exp_name += '_ema_d_'+ str(ema_decay)
    exp_name += '_m_consis_'+ str(mixup_consistency)
    exp_name += '_type_'+ str(consistency_type)
    exp_name += '_ramp_'+ str(consistency_rampup_s)
    exp_name += '_'+ str(consistency_rampup_e)
    
    
    exp_name += '_l2_'+str(l2)
    exp_name += '_eph_'+str(epochs)
    exp_name += '_bs_'+str(batch_size)
    
    if mixup_sup_alpha:
        exp_name += '_m_sup_a'+str(mixup_sup_alpha)
    if mixup_usup_alpha:
        exp_name += '_m_usup_a'+str(mixup_usup_alpha)
    if mixup_hidden :
        exp_name += 'm_hidden_'
        exp_name += str(num_mix_layer)
    exp_name += '_pl_'+str(pseudo_label)
    if nesterov:
        exp_name += '_nesterov_'
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)

    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name

def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    #x, y = x.numpy(), y.numpy()
    #mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index,:]
    #y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y = x.data.cpu().numpy(), y.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])
    
    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    return mixed_x, mixed_y, lam


import torch
import joblib
import os
import torchvision
import numpy as np
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.segmentation import mark_boundaries
from PIL import Image

def extract_features_and_save_to_csv(model, dataloader, output_file, batch_size=32):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    features_list = []
    labels_list = []
    model.eval()
    def hook(module, input, output):
       
        features_list.append(output.detach().cpu().numpy())
        print(features_list)

    
    handle = model.densenet121.classifier[0].register_forward_hook(hook)

    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            _ = model(inputs)  # عبور داده‌ها از مدل
            #print(features_list)
            labels_list.extend(labels.cpu().numpy()) 
            
            
            if (i + 1) % batch_size == 0:
                save_batch_to_csv(features_list, labels_list, output_file)
                features_list = []
                labels_list = []

    # ذخیره داده‌های باقی‌مانده
    if features_list:
        save_batch_to_csv(features_list, labels_list, output_file)

    # لغو ثبت hook
    handle.remove()

def save_batch_to_csv(features_list, labels_list, output_file):
    features_array = np.concatenate(features_list, axis=0)
    data = pd.DataFrame(features_array)
    data['label'] = labels_list
    data.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range    

def main():
    

    set_all_seeds(10)
    global global_step
    global best_prec1
    global best_test_ema_prec1
    global best_acc
    global best_accun
    
    print('| Building net type [' + args.arch + ']...')
    model = getNetwork(args, num_classes)
    ema_model = getNetwork(args, num_classes,ema=True)
    
    if use_cuda:
        model.cuda()
        ema_model.cuda()
        cudnn.benchmark = True

    if args.dataset != 'mnist':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum)
    exp_name = experiment_name(sl = args.sl,
                    dataset= args.dataset,
                    labels = args.num_labeled,
                    valid = args.num_valid_samples,
                    optimizer = args.optimizer,
                    lr = args.lr,
                    init_lr = args.initial_lr,
                    lr_rampup = args.lr_rampup, 
                    lr_rampdown = args.lr_rampdown_epochs,
                    l2 = args.weight_decay,
                    ema_decay = args.ema_decay, 
                    mixup_consistency = args.mixup_consistency,
                    consistency_type = args.consistency_type,
                    consistency_rampup_s = args.consistency_rampup_starts,
                    consistency_rampup_e = args.consistency_rampup_ends,
                    epochs = args.epochs,
                    batch_size = args.batch_size,
                    mixup_sup_alpha = args.mixup_sup_alpha,
                    mixup_usup_alpha = args.mixup_usup_alpha,
                    mixup_hidden = args.mixup_hidden,
                    num_mix_layer = args.num_mix_layer,
                    pseudo_label = args.pseudo_label,
                    arch = args.arch,
                    dropout = args.dropout,
                    nesterov = args.nesterov,
                    job_id = args.job_id,
                    add_name= args.add_name)
    exp_name='s'


    exp_dir = args.root_dir+exp_name
    print (exp_dir)
    import os
    if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    
    result_path = os.path.join(exp_dir , 'out.txt')
    result_pathun = os.path.join(exp_dir , 'outun.txt')
    result_pathc = os.path.join(exp_dir , 'outlossc.txt')    
    result_pathg = os.path.join(exp_dir , 'outlossg.txt')  
    if args.resume=='y':
        filep = open(result_path, 'a')
        filep2 = open(result_pathun, 'a')
        filepc = open(result_pathc, 'a')
        filepg = open(result_pathg, 'a')        
    else:
        filep = open(result_path, 'w')   
        filep2 = open(result_pathun, 'w')   
        filepc = open(result_pathc, 'w') 
        filepg = open(result_pathg, 'w')                         
    out_str = str(args)
    filep.write(out_str + '\n')  
    filep2.write(out_str + '\n')  
    filepc.write(out_str + '\n')         
    filepg.write(out_str + '\n')   
   
    
    if args.evaluate:
        print("Evaluating the primary model:\n")
        validate(validloader, model, global_step, args.start_epoch, filep)
        print("Evaluating the EMA model:\n")
        validate(validloader, ema_model, global_step, args.start_epoch, filep)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.sl:
            train_sl(trainloader, model, optimizer, epoch, filep)
        else:
            train(trainloader, unlabelledloader, model, ema_model, optimizer, epoch, filep,filep2,filepg,filepc)
        print("--- training epoch in %s seconds ---\n" % (time.time() - start_time))
        if args.pseudo_label == 'single':
                    print("Evaluating the primary model on test set:\n")
                    acc = validate(testloader, model, global_step, epoch + 1, filep, testing = True)
                    if acc > best_acc:
                        best_acc = acc
                        save(model,exp_dir, epoch + 1)
                    filep.write("Test error on the model with best validation error %s\n" % (best_acc.item()))
    
        else:
                    print("Evaluating the EMA model on test set:\n")
                    acc = validate(testloader, ema_model, global_step, epoch + 1, filep, ema= True, testing = True)        
                    if acc > best_acc:
                        best_acc = acc
                        save(ema_model,exp_dir, epoch + 1)


###########
##############
##############                                                
                        from sklearn.metrics import confusion_matrix
                        import seaborn as sn
                        import pandas as pd
                        import matplotlib.pyplot as plt

                        y_pred = []
                        y_true = []

                        # iterate over test data
                        for inputs, labels in testloader:
                                inputs, labels = inputs.cuda(), labels.cuda()

                                output = ema_model(inputs) # Feed Network

                                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                                y_pred.extend(output) # Save Prediction
                                
                                labels = labels.data.cpu().numpy()
                                y_true.extend(labels) # Save Truth

                        # constant for classes
                        classes = ('0', '1', '2','3','4','5')

                        # Build confusion matrix
                        cf_matrix = confusion_matrix(y_true, y_pred)
                        #df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                                            # columns = [i for i in classes])
                        df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                                            columns = [i for i in classes])                     
                        plt.figure(figsize = (12,7))
                        sn.heatmap(df_cm, annot=True,fmt="d", cmap='Greys',cbar=True)
                        plt.xlabel("Predicted lable")
                        plt.ylabel("True label (ground truth)")
                        plt.savefig('output.png')
                        plt.savefig('output.pdf')###
###
###
###
                        extract_features_and_save_to_csv(model, unlabelledloadera,'unlabled-features-tsne.csv')

                        import numpy as np
                        from sklearn.manifold import TSNE
                        df = pd.read_csv('unlabled-features-tsne.csv')

                        X = df.drop(columns=['label'])
                        y = df['label']
                        tsne = TSNE(n_components=2).fit_transform(X)

                        tx = tsne[:, 0]
                        ty = tsne[:, 1]

                        tx = scale_to_01_range(tx)
                        ty = scale_to_01_range(ty)

                        label_names = {0: 'Actinic Keratosis', 1: 'Basal Cell Carcinoma', 2: 'Melanoma',3:'Nevus',4:'Squamous Cell Carcinoma',5:'Seborrheic Keratosis'}
                        unique_labels = np.unique(y)

                        # Define colors per class (example)
                        unique_labels = np.unique(y)
                        colors_per_class = {label: plt.cm.viridis(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                        colors_per_class = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}  # Use tab10 for better contrast

                        fig = plt.figure(figsize=(15, 8))  # Adjust size as needed
                        ax = fig.add_subplot(111)

                        # Scatter plot for each class
                        for label in unique_labels:
                            indices = [i for i, l in enumerate(y) if l == label]
                            current_tx = np.take(tx, indices)
                            current_ty = np.take(ty, indices)
                            color = colors_per_class[label]
                            ax.scatter(current_tx, current_ty, c=color, label=label_names[label], s=50, alpha=0.7) 

                        ax.legend(loc='best')
                        plt.title("t-SNE Visualization of Features for  PAD-UFES  unlabeled images using %03d" % (args.num_labeled*num_classes)+" labeled images, additional unlabeled images, and informative fake images")
                        plt.xlabel("t-SNE Component 1")
                        plt.ylabel("t-SNE Component 2")
                        #plt.show()
                        generatedreportdr = 'experiments/s/' + args.dataset+'/tsneun/'
                        if not os.path.exists(generatedreportdr):
                            os.makedirs(generatedreportdr)
                        plt.savefig(generatedreportdr+'unlabled-tsne' + 'epoch%03d' % (epoch+1) +'.png')
                        plt.savefig(generatedreportdr+'unlabled-tsne'+ 'epoch%03d' % (epoch+1) +'.pdf')
                        import os

####
                        for i in range(50)  :  
                            with torch.no_grad():
                                        visualize_resultscreat(G, (i + 1))
                        from torchvision import datasets, transforms
                        from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
                        import numpy as np                                        
                        fake_data = torchvision.datasets.ImageFolder('skinfake/', transforms.ToTensor())
                        subset = np.random.permutation([i for i in range(len(fake_data))])
                        print (subset.shape)
                        sampler_fake = SubsetRandomSampler(subset)
                        fakeloader = torch.utils.data.DataLoader(fake_data, batch_size=32,sampler=sampler_fake, num_workers=1, pin_memory=True)
                        #print('salam')
                        extract_features_and_save_to_csv(model, fakeloader,'fake-features-tsne.csv')





                        #extract_features_and_save_to_csv(model, trainloader,'unlabled-features-tsne.csv')
                        import pandas as pd

                        # Load both CSV files (assuming they have headers)
                        df_fake = pd.read_csv('fake-features-tsne.csv')  # First file
                        df_unlabeled = pd.read_csv('unlabled-features-tsne.csv')  # Second file

                        # Step 1: Set last column of first CSV to 3
                        last_col_name = df_fake.columns[-1]  # Get the last column's name (e.g., 'class')
                        df_fake[last_col_name] = 6  # Overwrite all values in this column with 3

                        # Step 2: Combine DataFrames vertically (preserving headers)
                        combined_df = pd.concat([df_fake, df_unlabeled], axis=0, ignore_index=True)

                        # Step 3: Save with header
                        combined_df.to_csv('combined-features-tsne.csv', index=False)







                        import numpy as np
                        from sklearn.manifold import TSNE
                        df = pd.read_csv('combined-features-tsne.csv')

                        X = df.drop(columns=['label'])
                        y = df['label']
                        tsne = TSNE(n_components=2).fit_transform(X)

                        tx = tsne[:, 0]
                        ty = tsne[:, 1]

                        tx = scale_to_01_range(tx)
                        ty = scale_to_01_range(ty)

                        label_names = {0: 'Actinic Keratosis', 1: 'Basal Cell Carcinoma', 2: 'Melanoma',3:'Nevus',4:'Squamous Cell Carcinoma',5:'Seborrheic Keratosis',6:'Generated support vectors '}  # Modify with your actual class names
                        unique_labels = np.unique(y)

                        # Define colors per class (example)
                        unique_labels = np.unique(y)
                        colors_per_class = {label: plt.cm.viridis(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                        colors_per_class = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}  # Use tab10 for better contrast

                        fig = plt.figure(figsize=(15, 8))  # Adjust size as needed
                        ax = fig.add_subplot(111)

                        # Scatter plot for each class
                        for label in unique_labels:
                            indices = [i for i, l in enumerate(y) if l == label]
                            current_tx = np.take(tx, indices)
                            current_ty = np.take(ty, indices)
                            color = colors_per_class[label]
                            ax.scatter(current_tx, current_ty, c=color, label=label_names[label], s=50, alpha=0.7) 

                        ax.legend(loc='best')
                        plt.title("t-SNE Visualization of Features for  PAD-UFES  generated support vectors and unlabeled images using %03d" % (args.num_labeled*num_classes)+" labeled images, additional unlabeled images, and informative fake images")
                        plt.xlabel("t-SNE Component 1")
                        plt.ylabel("t-SNE Component 2")
                        #plt.show()
                        generatedreportdr = 'experiments/s/' + args.dataset+'/tsnefakeunl/'
                        if not os.path.exists(generatedreportdr):
                            os.makedirs(generatedreportdr)
                        plt.savefig(generatedreportdr+'fake-unlabled-tsne' + 'epoch%03d' % (epoch+1) +'.png')
                        plt.savefig(generatedreportdr+'fake-unlabled-tsne'+ 'epoch%03d' % (epoch+1)  +'.pdf')


##############
###
                        extract_features_and_save_to_csv(model, trainloader,'labled-features-tsne.csv')

                        import numpy as np
                        from sklearn.manifold import TSNE
                        df = pd.read_csv('labled-features-tsne.csv')

                        X = df.drop(columns=['label'])
                        y = df['label']
                        tsne = TSNE(n_components=2).fit_transform(X)

                        tx = tsne[:, 0]
                        ty = tsne[:, 1]

                        tx = scale_to_01_range(tx)
                        ty = scale_to_01_range(ty)

                        label_names = {0: 'Actinic Keratosis', 1: 'Basal Cell Carcinoma', 2: 'Melanoma',3:'Nevus',4:'Squamous Cell Carcinoma',5:'Seborrheic Keratosis'}                        
                        unique_labels = np.unique(y)

                        # Define colors per class (example)
                        unique_labels = np.unique(y)
                        colors_per_class = {label: plt.cm.viridis(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                        colors_per_class = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}  # Use tab10 for better contrast

                        fig = plt.figure(figsize=(15, 8))  # Adjust size as needed
                        ax = fig.add_subplot(111)

                        # Scatter plot for each class
                        for label in unique_labels:
                            indices = [i for i, l in enumerate(y) if l == label]
                            current_tx = np.take(tx, indices)
                            current_ty = np.take(ty, indices)
                            color = colors_per_class[label]
                            ax.scatter(current_tx, current_ty, c=color, label=label_names[label], s=50, alpha=0.7) 

                        ax.legend(loc='best')
                        plt.title("t-SNE Visualization of Features for  PAD-UFES  labeled images using %03d" % (args.num_labeled*num_classes)+" labeled images, additional unlabeled images, and informative fake images")
                        plt.xlabel("t-SNE Component 1")
                        plt.ylabel("t-SNE Component 2")
                        #plt.show()
                        generatedreportdr = 'experiments/s/' + args.dataset+'/tsnelabe/'
                        if not os.path.exists(generatedreportdr):
                            os.makedirs(generatedreportdr)
                        plt.savefig(generatedreportdr+'labled-tsne' + 'epoch%03d' % (epoch+1) +'.png')
                        plt.savefig(generatedreportdr+'labled-tsne'+ 'epoch%03d'  % (epoch+1) +'.pdf')

#####
##############
###





                        #extract_features_and_save_to_csv(model, trainloader,'unlabled-features-tsne.csv')
                        import pandas as pd

                        # Load both CSV files (assuming they have headers)
                        df_fake = pd.read_csv('fake-features-tsne.csv')  # First file
                        df_unlabeled = pd.read_csv('labled-features-tsne.csv')  # Second file

                        # Step 1: Set last column of first CSV to 3
                        last_col_name = df_fake.columns[-1]  # Get the last column's name (e.g., 'class')
                        df_fake[last_col_name] = 6  # Overwrite all values in this column with 3

                        # Step 2: Combine DataFrames vertically (preserving headers)
                        combined_df = pd.concat([df_fake, df_unlabeled], axis=0, ignore_index=True)

                        # Step 3: Save with header
                        combined_df.to_csv('combinedl-features-tsne.csv', index=False)







                        import numpy as np
                        from sklearn.manifold import TSNE
                        df = pd.read_csv('combinedl-features-tsne.csv')

                        X = df.drop(columns=['label'])
                        y = df['label']
                        tsne = TSNE(n_components=2).fit_transform(X)

                        tx = tsne[:, 0]
                        ty = tsne[:, 1]

                        tx = scale_to_01_range(tx)
                        ty = scale_to_01_range(ty)

                        label_names = {0: 'Actinic Keratosis', 1: 'Basal Cell Carcinoma', 2: 'Melanoma',3:'Nevus',4:'Squamous Cell Carcinoma',5:'Seborrheic Keratosis',6:'Generated support vectors '}  # Modify with your actual class names
                        unique_labels = np.unique(y)

                        # Define colors per class (example)
                        unique_labels = np.unique(y)
                        colors_per_class = {label: plt.cm.viridis(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                        colors_per_class = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}  # Use tab10 for better contrast

                        fig = plt.figure(figsize=(15, 8))  # Adjust size as needed
                        ax = fig.add_subplot(111)

                        # Scatter plot for each class
                        for label in unique_labels:
                            indices = [i for i, l in enumerate(y) if l == label]
                            current_tx = np.take(tx, indices)
                            current_ty = np.take(ty, indices)
                            color = colors_per_class[label]
                            ax.scatter(current_tx, current_ty, c=color, label=label_names[label], s=50, alpha=0.7) 

                        ax.legend(loc='best')
                        plt.title("t-SNE Visualization of Features for  PAD-UFES  generated support vectors and labeled images using %03d" % (args.num_labeled*num_classes)+" labeled images, additional unlabeled images, and informative fake images")
                        plt.xlabel("t-SNE Component 1")
                        plt.ylabel("t-SNE Component 2")
                        #plt.show()
                        generatedreportdr = 'experiments/s/' + args.dataset+'/tsnefakel/'
                        if not os.path.exists(generatedreportdr):
                            os.makedirs(generatedreportdr)
                        plt.savefig(generatedreportdr+'fake-labled-tsne' + 'epoch%03d' % (epoch+1) +'.png')
                        plt.savefig(generatedreportdr+'fake-labled-tsne'+ 'epoch%03d' % (epoch+1)  +'.pdf')


##############

                        extract_features_and_save_to_csv(model, testloader,'test-features-tsne.csv')

                        import numpy as np
                        from sklearn.manifold import TSNE
                        df = pd.read_csv('test-features-tsne.csv')

                        X = df.drop(columns=['label'])
                        y = df['label']
                        tsne = TSNE(n_components=2).fit_transform(X)

                        tx = tsne[:, 0]
                        ty = tsne[:, 1]

                        tx = scale_to_01_range(tx)
                        ty = scale_to_01_range(ty)

                        label_names = {0: 'Actinic Keratosis', 1: 'Basal Cell Carcinoma', 2: 'Melanoma',3:'Nevus',4:'Squamous Cell Carcinoma',5:'Seborrheic Keratosis'}
                        unique_labels = np.unique(y)

                        # Define colors per class (example)
                        unique_labels = np.unique(y)
                        colors_per_class = {label: plt.cm.viridis(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
                        colors_per_class = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}  # Use tab10 for better contrast

                        fig = plt.figure(figsize=(15, 8))  # Adjust size as needed
                        ax = fig.add_subplot(111)

                        # Scatter plot for each class
                        for label in unique_labels:
                            indices = [i for i, l in enumerate(y) if l == label]
                            current_tx = np.take(tx, indices)
                            current_ty = np.take(ty, indices)
                            color = colors_per_class[label]
                            ax.scatter(current_tx, current_ty, c=color, label=label_names[label], s=50, alpha=0.7) 

                        ax.legend(loc='best')
                        plt.title("t-SNE Visualization of Features for  PAD-UFES test images using %03d" % (args.num_labeled*num_classes)+" labeled images, additional unlabeled images, and informative fake images")
                        plt.xlabel("t-SNE Component 1")
                        plt.ylabel("t-SNE Component 2")
                        #plt.show()
                        generatedreportdr = 'experiments/s/' + args.dataset+'/tsnetest/'
                        if not os.path.exists(generatedreportdr):
                            os.makedirs(generatedreportdr)
                        plt.savefig(generatedreportdr+'test-tsne' + 'epoch%03d'  % (epoch+1)+'.png')
                        plt.savefig(generatedreportdr+'test-tsne'+ 'epoch%03d' % (epoch+1) +'.pdf')





















                        import os
                        os.remove("unlabled-features-tsne.csv") 
                        os.remove("fake-features-tsne.csv") 
                        os.remove("combined-features-tsne.csv")  
                        os.remove("combinedl-features-tsne.csv")   

                        os.remove("labled-features-tsne.csv")   
                        os.remove("test-features-tsne.csv")                  

                        

                        



                    filep.write("Test error on the model with best validation error %s\n" % (best_acc.item()))
                    acc = validate(unlabelledloader, ema_model, global_step, epoch + 1, filep, ema= True, testing = True)        
                    if acc > best_accun:
                        best_accun = acc
                    filep2.write("Test error on the model with best validation error %s\n" % (best_accun.item()))








        train_log = OrderedDict()
        train_log['train_class_loss_list'] = train_class_loss_list
        train_log['train_ema_class_loss_list'] = train_ema_class_loss_list
        train_log['train_mixup_consistency_loss_list'] = train_mixup_consistency_loss_list
        train_log['train_mixup_consistency_coeff_list'] = train_mixup_consistency_coeff_list
        train_log['train_error_list'] = train_error_list
        train_log['train_ema_error_list'] = train_ema_error_list
        train_log['train_lr_list'] = train_lr_list
        train_log['val_class_loss_list'] = val_class_loss_list
        train_log['val_error_list'] = val_error_list
        train_log['val_ema_class_loss_list'] = val_ema_class_loss_list
        train_log['val_ema_error_list'] = val_ema_error_list
        
        filep.flush()
        filep2.flush()
        filepg.flush()
        filepc.flush()
        pickle.dump(train_log, open( os.path.join(exp_dir,'log.pkl'), 'wb'))

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
    from imblearn.metrics import specificity_score,sensitivity_score,geometric_mean_score,classification_report_imbalanced
    print('full specificity_score',specificity_score(y_true, y_pred, average='weighted'))
    print('any specificity_score',specificity_score(y_true, y_pred, average=None))
    print('full sensitivity_score',sensitivity_score(y_true, y_pred, average='weighted'))
    print('any sensitivity_score',sensitivity_score(y_true, y_pred, average=None))
    print('full geometric_mean_score',geometric_mean_score(y_true, y_pred, average='weighted'))
    print('any geometric_mean_score',geometric_mean_score(y_true, y_pred, average=None))
    print(classification_report_imbalanced(y_true, y_pred,     target_names=classes,digits=6))
    print(accuracy_score(y_true, y_pred))


        
def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train_sl(trainloader, model, optimizer, epoch, filep):
    global global_step
    
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    
    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    
    end = time.time()
    
    for i, (input, target)in enumerate(trainloader):
        # measure data loading time
        meters.update('data_time', time.time() - end)
        #if args.dataset == 'cifar10':
            #input = apply_zca(input, zca_mean, zca_components)
        
        
        lr = adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
        meters.update('lr', optimizer.param_groups[0]['lr'])
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.cuda())

        minibatch_size = len(target_var)
        #labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        #assert labeled_minibatch_size > 0
        
        model_out = model(input_var)

        logit1 = model_out
        class_logit, cons_logit = logit1, logit1
        
        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        meters.update('class_loss', class_loss.item())

        
        loss = class_loss
        #assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.data[0])
        assert not (np.isnan(loss.item())), 'Loss explosion: {}'.format(loss.data[0])
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 6))
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100. - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100. - prec5[0], minibatch_size)

        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()



            
    train_class_loss_list.append(meters['class_loss'].avg)
    train_error_list.append(meters['error1'].avg)
    train_lr_list.append(meters['lr'].avg)
    


def train(trainloader,unlabelledloader, model, ema_model, optimizer, epoch, filep,filep2,filepg,filepc):
    global global_step
    
    class_criterion = nn.CrossEntropyLoss().cuda()
    criterion_u= nn.KLDivLoss(reduction='batchmean').cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    
    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    i = -1

    labeled_iter = trainloader.__iter__()
    for o,(u, _) in enumerate(unlabelledloader):
        # measure data loading time
        i = i+1
        try:
              (input, target) = labeled_iter.__next__()
        except StopIteration:
              labeled_iter = trainloader.__iter__()
              (input, target)= labeled_iter.__next__()

            
        meters.update('data_time', time.time() - end)
        #print(input.shape)
        if input.shape[0]!= u.shape[0]:
            bt_size = np.minimum(input.shape[0], u.shape[0])
            input = input[0:bt_size]
            target = target[0:bt_size]
            u = u[0:bt_size]
        
        
        #if args.dataset == 'cifar10':
            #input = apply_zca(input, zca_mean, zca_components)
            #u = apply_zca(u, zca_mean, zca_components) 
        if args.dataset != 'mnist':
            lr = adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
            meters.update('lr', optimizer.param_groups[0]['lr'])
        
        if args.mixup_sup_alpha:
            if use_cuda:
                input , target, u  = input.cuda(), target.cuda(), u.cuda()
            input_var, target_var, u_var = Variable(input), Variable(target), Variable(u) 
            
            if args.mixup_hidden:
                output_mixed_l, target_a_var, target_b_var, lam = model(input_var, target_var, mixup_hidden = True,  mixup_alpha = args.mixup_sup_alpha, layers_mix = args.num_mix_layer)
                lam = lam[0]
            else:
                mixed_input, target_a, target_b, lam = mixup_data_sup(input, target, args.mixup_sup_alpha)
                #if use_cuda:
                #    mixed_input, target_a, target_b  = mixed_input.cuda(), target_a.cuda(), target_b.cuda()
                mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
                output_mixed_l = model(mixed_input_var)
                #print(output_mixed_l.shape)    
            loss_func = mixup_criterion(target_a_var, target_b_var, lam)
            class_loss = loss_func(class_criterion, output_mixed_l)
            
        else:
            input_var = torch.autograd.Variable(input.cuda())
            with torch.no_grad():
                u_var = torch.autograd.Variable(u.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            output = model(input_var)
            class_loss = class_criterion(output, target_var)
        
        meters.update('class_loss', class_loss.item())
        
        ### get ema loss. We use the actual samples(not the mixed up samples ) for calculating EMA loss
        minibatch_size = len(target_var)
        if args.pseudo_label == 'single':
            ema_logit_unlabeled = model(u_var)
            ema_logit_labeled = model(input_var)
        else:
            ema_logit_unlabeled = ema_model(u_var)
            ema_logit_labeled = ema_model(input_var)
        if args.mixup_sup_alpha:
            class_logit = model(input_var)
        else:
            class_logit = output
        cons_logit = model(u_var)

        ema_logit_unlabeled = Variable(ema_logit_unlabeled.detach().data, requires_grad=False)

        #class_loss = class_criterion(class_logit, target_var) / minibatch_size
        
        ema_class_loss = class_criterion(ema_logit_labeled, target_var)# / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())
        
               
        ### get the unsupervised mixup loss###
        if args.mixup_consistency:
                if args.mixup_hidden:
                    #output_u = model(u_var)
                    output_mixed_u, target_a_var, target_b_var, lam = model(u_var, ema_logit_unlabeled, mixup_hidden = True,  mixup_alpha = args.mixup_sup_alpha, layers_mix = args.num_mix_layer)
                    # ema_logit_unlabeled
                    lam = lam[0]
                    mixedup_target = lam * target_a_var + (1 - lam) * target_b_var
                else:
                    #output_u = model(u_var)
                    mixedup_x, mixedup_target, lam = mixup_data(u_var, ema_logit_unlabeled, args.mixup_usup_alpha)
                    #mixedup_x, mixedup_target, lam = mixup_data(u_var, output_u, args.mixup_usup_alpha)
                    output_mixed_u = model(mixedup_x)
                mixup_consistency_loss = consistency_criterion(output_mixed_u, mixedup_target) / minibatch_size# criterion_u(F.log_softmax(output_mixed_u,1), F.softmax(mixedup_target,1))
                meters.update('mixup_cons_loss', mixup_consistency_loss.item())
                if epoch < args.consistency_rampup_starts:
                    mixup_consistency_weight = 0.0
                else:
                    mixup_consistency_weight = get_current_consistency_weight(args.mixup_consistency, epoch, i, len(unlabelledloader))
                meters.update('mixup_cons_weight', mixup_consistency_weight)
                mixup_consistency_loss = mixup_consistency_weight*mixup_consistency_loss
        else:
            mixup_consistency_loss = 0
            meters.update('mixup_cons_loss', 0)
        
        #labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        #assert labeled_minibatch_size > 0





        
        #labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        #assert labeled_minibatch_size > 0










        C_fake_pred= model(u_var)
        C_fake_pred = F.softmax(C_fake_pred, dim=1)










        if args.dataset == 'mnist':
            generated_batch_size=64
        else:    
            generated_batch_size=32
        z_ = torch.randn((generated_batch_size, z_dim))
        z_ = z_.cuda()
        G_ = G(z_)






        G_ = G(z_)
        C_fake_pred= model(G_)
        C_fake_pred = F.softmax(C_fake_pred, dim=1)
        C_fake_wei = torch.max(C_fake_pred, 1)[1]                              
        C_fake_wei = C_fake_wei.view(-1, 1)
        C_fake_wei = torch.zeros(C_fake_wei.size(0), num_classes).cuda().scatter_(1, C_fake_wei, 1)


        C_fake_loss = nll_loss_neg(C_fake_wei, C_fake_pred,wcw2)

        #d3,l3 = mixup_batch2(mixup,G_,C_fake_wei)
        #outputsm= model(d3)
        #outputsm = F.softmax(outputsm, dim=1)
        #outputsm = torch.max(outputsm, 1)[1]
        #outputsm = outputsm.view(-1, 1)
        #outputsm = torch.zeros(outputsm.size(0), 10).cuda().scatter_(1, outputsm, 1)               
        #C_fake_lossmix = nll_loss_neg(outputsm, l3)




















        C_fake_pred= model(u_var)
        C_fake_pred = F.softmax(C_fake_pred, dim=1)


        C_fake_wei = torch.max(C_fake_pred, 1)[1]                              
        #C_fake_wei = C_fake_wei.view(-1, 1)
        #C_fake_wei = torch.zeros(C_fake_wei.size(0), num_classes).cuda().scatter_(1, C_fake_wei, 1)
        C_unlabeled_loss = F.nll_loss(C_fake_pred,C_fake_wei,wcw2)


        output = model(input_var)
        #output=F.log_softmax(output, dim=1)            
        C_labeled_loss = F.nll_loss(output, target_var)







        #generated_weight(epoch) * C_fake_loss        
        loss = 0.1* C_fake_loss+C_unlabeled_loss +class_loss+mixup_consistency_loss#C_labeled_loss+0.1*C_fake_lossmix+C_labeled_loss +C_unlabeled_loss+generated_weight(epoch)*C_u_lossmix
        meters.update('loss', loss.item())



        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1,6))
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100. - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100. - prec5[0], minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit_labeled.data, target_var.data, topk=(1, 2))
        meters.update('ema_top1', ema_prec1[0], minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], minibatch_size)
        meters.update('ema_top5', ema_prec5[0], minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5[0], minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Mixup Cons {meters[mixup_cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(unlabelledloader), meters=meters))
            #print ('lr:',optimizer.param_groups[0]['lr'])




        


        
        # update D network
        #D_optimizer.zero_grad()
        #G_ = G(z_)
        #d2,l2 = mixup_batch(mixup,input_var,u_var,G_)                   
        #D_loss_mix = BCEloss(D(d2), l2)
        #D_loss = D_loss_mix#+D_labeled_loss + D_unlabeled_loss + D_fake_loss
        #D_loss.backward()
        #D_optimizer.step()


                # update D network
        if True:        
                D_optimizer.zero_grad()

                D_labeled = D(input_var)
                D_labeled_loss = BCEloss(D_labeled, torch.ones_like(D_labeled))
                acl=D.features

                D_unlabeled = D(u_var)
                D_unlabeled_loss = BCEloss(D_unlabeled, torch.ones_like(D_unlabeled))
                acul=D.features
                G_ = G(z_)
                D_fake = D(G_.detach())
                actdf=D.features           
                D_fake_loss = BCEloss(D_fake, torch.zeros_like(D_fake)  )

                T_x = transform(G_.detach())
                D_faket = D(T_x)
                L_D_fake1 = l2loss(D_fake , D_faket )


                T_x = transform(input_var)
                D_labeledt = D(T_x)
                L_reall1 = l2loss(D_labeled , D_labeledt )
                actl=D.features
                consistency_relation_distl = torch.sum(losses2.relation_mse_loss(acl, actl)) / D_labeledt.size(0)


                T_x = transform(u_var)
                D_unlabeledt = D(T_x)
                L_realu2 = l2loss(D_unlabeled , D_unlabeledt )

                actul=D.features
                D_fake_loss = BCEloss(D_fake, torch.zeros_like(D_fake)  )
                consistency_relation_distul = torch.sum(losses2.relation_mse_loss(acul, actul)) / D_unlabeledt.size(0)


                T_z = z_ + torch.normal(0,1,z_.shape).cuda()
                G_T_z = G(T_z)
                D_G_T_z = D(G_T_z.detach())
                # zCR: Calculate L_dis |D(G(z)) ? D(G(T(z))|^2
                actdgtz=D.features
                L_dis = l2loss(D_fake, D_G_T_z)
                consistency_relation_distdf = torch.sum(losses2.relation_mse_loss(actdgtz, actdf)) / D_G_T_z.size(0)

                D_loss = D_labeled_loss + D_unlabeled_loss + D_fake_loss+L_dis*0.1+consistency_relation_distdf*0.1#+((L_reall1+L_realu2+L_D_fake1)*0.1)+consistency_relation_distl*0.1+consistency_relation_distul*0.1#self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                D_optimizer.step()

        # update G network
        #G_optimizer.zero_grad()


        #G_ = G(z_)
        #D_fake = D(G_)








        #d,l = mixup_batch(0,input_var,u_var,G_)               
        #G_loss_mix = -BCEloss(D(d), l)






        #C_fake_pred= model(G_)

        #C_fake_pred = F.log_softmax(C_fake_pred, dim=1)
        #with torch.no_grad():
            #C_fake_wei = torch.max(C_fake_pred, 1)[1]
        #G_loss_C = F.nll_loss(C_fake_pred, C_fake_wei)















        #G_ = G(z_)
        #C_fake_pred= model(G_)
        #C_fake_pred = F.softmax(C_fake_pred, dim=1)
        #C_fake_wei = torch.max(C_fake_pred, 1)[1]                              
        #C_fake_wei = C_fake_wei.view(-1, 1)
        #C_fake_wei = torch.zeros(C_fake_wei.size(0), num_classes).cuda().scatter_(1, C_fake_wei, 1)

        #d3,l3 = mixup_batch2(mixup,G_,C_fake_wei)
        #outputsm= model(d3)
        #outputsm = F.softmax(outputsm, dim=1)
        
        #outputsm = torch.max(outputsm, 1)[1]
        #outputsm = outputsm.view(-1, 1)
        #outputsm = torch.zeros(outputsm.size(0), num_classes).cuda().scatter_(1, outputsm, 1)               
        #G_C_fake_lossix = nll_loss_neg2(outputsm, l3)











        #if epoch <= 3:
         #   G_loss_mix.backward()
        #else:
          #  G_loss_mix.backward(retain_graph=True)
          #  G_loss_C.backward()
           # G_C_fake_lossix.backward()

        #G_loss = G_loss_D + generated_weight(epoch) * G_loss_C+mixup_consistency_lossf
        #G_loss=G_loss_mix+ 0.1 * G_loss_C+0.1*G_C_fake_lossix

        #G_loss.backward()

        #G_optimizer.step()


        if True:
                    # update G network
                G_optimizer.zero_grad()

                G_ = G(z_)
                T_z = z_ + torch.normal(0,1,z_.shape).cuda()
                G_T_z = G(T_z)


                D_fake = D(G_)
                G_loss_D = BCEloss(D_fake, torch.ones_like(D_fake))

                consistency_relation_distfg = -torch.sum(losses2.relation_mse_loss(G_, G_T_z)) / G_.size(0)


                L_gen = -l2loss(G_, G_T_z)

                #t1,_=self.C(G_)
                #t2,_=self.C(G_T_z)

                #L_genC = -l2loss(t1, t2)







                C_fake_pred = model(G_)
                C_fake_wei = torch.max(C_fake_pred, 1)[1]
                
                G_loss_C  = F.nll_loss(C_fake_pred, C_fake_wei,wcw2)


                #_, C_fake_pred = self.C(G_T_z)
                #C_fake_wei = torch.max(C_fake_pred, 1)[1]
                #G_loss_Ct  = F.nll_loss(C_fake_pred, C_fake_wei)





                G_loss = G_loss_D + G_loss_C+L_gen*0.1+consistency_relation_distfg*0.1
                #self.train_hist['G_loss'].append(G_loss.item())

                #G_loss_D.backward(retain_graph=True)
                #G_loss_C.backward()
                G_loss.backward()
                G_optimizer.step()
    #filepg.write("Test error on the model with best validation error %s\n" % (G_loss.item()))
    # filepc.write("Test error on the model with best validation error %s\n" % (loss.item()))
    filepg.write("Test error on the model with best validation error %s\n" % (G_loss_C.item()))
    filepc.write("Test error on the model with best validation error %s\n" % (0.1*C_fake_loss.item()))

            






def visualize_results(G, epoch):
    G.eval()
    generated_images_dir = 'experiments/s/' + args.dataset
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)



    tot_num_samples = 64
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    sample_z_ = torch.randn((tot_num_samples, z_dim))

    sample_z_ = sample_z_.cuda()

    samples = G(sample_z_)

    samples1=samples
    samples = samples.mul(0.5).add(0.5)

    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)


    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      generated_images_dir + '/' + 'epoch%03d' % epoch + '.png')
    gridOfFakeImages = torchvision.utils.make_grid((samples1 + 1) / 2)
    torchvision.utils.save_image(gridOfFakeImages,generated_images_dir + '/' + 'epoch%03d' % epoch + '.pdf')


    

def visualize_resultscreat(G, epoch):
    G.eval()
    generated_images_dir = 'skinfake/1/' 
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

    tot_num_samples = 1
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    sample_z_ = torch.randn((tot_num_samples, z_dim))

    sample_z_ = sample_z_.cuda()

    samples = G(sample_z_)

    samples1=samples
    samples = samples.mul(0.5).add(0.5)

    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)


    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      generated_images_dir + '/' + 'epoch%03d' % epoch + '.png')
    gridOfFakeImages = torchvision.utils.make_grid((samples1 + 1) / 2)
    #torchvision.utils.save_image(gridOfFakeImages,generated_images_dir + '/' + 'epoch%03d' % epoch + '.pdf')

  

def validate(eval_loader, model, global_step, epoch, filep, ema = False, testing = False):
    #from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
    #from imblearn.metrics import specificity_score,sensitivity_score,geometric_mean_score,classification_report_imbalanced        

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()
    with torch.no_grad():
                visualize_results(G, (epoch + 1))
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)
    
        #if args.dataset == 'cifar10':
            #input = apply_zca(input, zca_mean, zca_components)
            
        with torch.no_grad():        
            input_var = torch.autograd.Variable(input.cuda())
        with torch.no_grad():
            target_var = torch.autograd.Variable(target.cuda())

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        output1 = model(input_var)
        softmax1 = F.softmax(output1, dim=1)
        class_loss = class_criterion(output1, target_var) / minibatch_size
        #prec1=f1_score(target_var.data, output1.data, average='weighted')
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 6))
        meters.update('class_loss', class_loss.item(), minibatch_size)
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100.0 - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100.0 - prec5[0], minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()
        
    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'
          .format(top1=meters['top1'], top5=meters['top5']))

    
    if testing == False:
        if ema:
            val_ema_class_loss_list.append(meters['class_loss'].avg)
            val_ema_error_list.append(meters['error1'].avg)
        else:
            val_class_loss_list.append(meters['class_loss'].avg)
            val_error_list.append(meters['error1'].avg)
    
    
    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    print("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print
        ("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

def adjust_learning_rate_step(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch, total_steps_in_epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - args.consistency_rampup_starts
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    #labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8).type(torch.cuda.FloatTensor)
    minibatch_size = len(target)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / minibatch_size))
    return res


def save(cc,save_dir, ep):

    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    torch.save(G.state_dict(), os.path.join(save_dir, '_G.pkl'))
    torch.save(D.state_dict(), os.path.join(save_dir, '_D.pkl'))
    torch.save(cc.state_dict(), os.path.join(save_dir,  '_C.pkl'))

if __name__ == '__main__':
     main()
