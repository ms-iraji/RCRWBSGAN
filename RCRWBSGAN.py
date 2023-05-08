
import logging

import re
import argparse
import os
import shutil
import time
import math

import utils2, torch, time, os, pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets


import torchvision.transforms as transforms
import torchvision

transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                transforms.RandomAffine(0, (1/8,0))]) # max horizontal shift by 4
l2loss = nn.MSELoss()

CLASS_NUM2 = [1113, 6705, 514, 327, 1099, 115, 142]
CLASS_WEIGHT2 = torch.Tensor([10000/i for i in CLASS_NUM2]).cuda()
min1=torch.min(CLASS_WEIGHT2)

max1=torch.max(CLASS_WEIGHT2) 
cw=(CLASS_WEIGHT2-min1)
wcw2 = (cw / (((max1.clone().detach().item())-min1.clone().detach().item())))* (0.9 - 0.5) + 0.5
print(wcw2)

def nll_loss_neg(y_pred, y_true,CLASS_WEIGHT):
    out = torch.sum((y_true * y_pred)*CLASS_WEIGHT, dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))

def nll_loss_neg2(y_pred, y_true):
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log(( out) + 1e-6))


class generator(nn.Module):  # # #
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
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 4,2,1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils2.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):  # # #
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
        utils2.initialize_weights(self)
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







import os
import sys
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.models import DenseNet121
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import  dataset
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics,epochVal_metrics_test


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/content/ISIC2018_Task3_Training_Input/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/content/data/skin/t.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/content/data/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/content/data/skin/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--epochs', type=int,  default=60, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=36, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=12, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_num', type=int, default=1400, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--consistency_relation_weight', type=int,  default=1, help='consistency relation weight')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
args = parser.parse_args()






def visualize_results(G, epoch):
    G.eval()
    generated_images_dir = 'generated_images222/' 
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

    tot_num_samples = 64
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    sample_z_ = torch.rand((tot_num_samples, args.z_dim))

    sample_z_ = sample_z_.cuda()

    samples = G(sample_z_)

    samples1=samples
    samples = samples.mul(0.5).add(0.5)

    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
    #samples = (samples + 1) / 2

    utils2.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      generated_images_dir + '/' + 'epoch%03d' % epoch + '.png')
    gridOfFakeImages = torchvision.utils.make_grid((samples1 + 1) / 2)
    torchvision.utils.save_image(gridOfFakeImages,generated_images_dir + '/' + 'epoch%03d' % epoch + '.pdf')


train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


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




def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
                                 betas=(0.9, 0.999), weight_decay=5e-4)

    args.generated_batch_size=32
    args.z_dim=100
    G = generator(input_dim=args.z_dim, output_dim=3, input_size=112)
    D = discriminator(input_dim=3, output_dim=1, input_size=112)

    G.cuda()
    D.cuda()
    args.lrG=0.0002
    args.lrD=0.0002
    args.beta1=0.5
    args.beta2=0.999
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    BCEloss = nn.BCELoss().cuda()




    # dataset
    normalize = transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])


    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.Resize((112, 112)),
                                                #transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                #transforms.RandomHorizontalFlip(),
                                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                # transforms.RandomRotation(10),
                                                # transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])))

    train_dataset2 = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=transforms.Compose([
                                                transforms.Resize((112, 112)),
                                                #transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                #transforms.RandomHorizontalFlip(),
                                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                # transforms.RandomRotation(10),
                                                # transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ]))

    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((112, 112)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_test,
                                          transform=transforms.Compose([
                                              transforms.Resize((112, 112)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 10000))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)#, worker_init_fn=worker_init_fn)
    train_dataloaderu = DataLoader(dataset=train_dataset2, batch_size=64)
           
    model.train()

    loss_fn = losses.cross_entropy_loss()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    model.train()
    D.train()
    G.train()
    #train
    for epoch in range(1, 3):

      j=0    
      for o, (_,_, (image_batchl, ema_image_batchl), label_batchl) in enumerate(train_dataloader):
        j=j+1
    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()
        iter_max = len(train_dataloader)

        labeled_iter = train_dataloader.__iter__()
        for i, (_,_, image_batch, label_batch) in enumerate(train_dataloaderu):
            time2 = time.time()
            try:
              (_,_, (image_batchl, ema_image_batchl), label_batchl) = labeled_iter.__next__()
            except StopIteration:
              labeled_iter = train_dataloader.__iter__()
              (_,_, (image_batchl, ema_image_batchl), label_batchl)= labeled_iter.__next__()
            # print('fetch data cost {}'.format(time2-time1))
            image_batchl, ema_image_batchl, label_batchl = image_batchl.cuda(), ema_image_batchl.cuda(), label_batchl.cuda()
            # unlabeled_image_batch = ema_image_batch[labeled_bs:]

            # noise1 = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.1, 0.1)
            # noise2 = torch.clamp(torch.randn_like(ema_image_batch) * 0.1, -0.1, 0.1)
            ema_inputsl = ema_image_batchl #+ noise2
            inputsl = image_batchl #+ noise1
            




            activationsl, outputsl = model(inputsl)
            with torch.no_grad():
                ema_activationsl, ema_outputl = ema_model(ema_inputsl)

            
            ## calculate the loss
            loss_classification = loss_fn(outputsl[:labeled_bs], label_batchl[:labeled_bs])
            loss = loss_classification


            if args.ema_consistency == 1:
                consistency_weight = get_current_consistency_weight(epoch)
                consistency_dist = torch.sum(losses.softmax_mse_loss(outputsl, ema_outputl)) / batch_size #/ dataset.N_CLASSES
                consistency_loss = consistency_weight * consistency_dist  

                # consistency_relation_dist = torch.sum(losses.relation_mse_loss_cam(activations, ema_activations, model, label_batch)) / batch_size
                consistency_relation_dist = torch.sum(losses.relation_mse_loss(activationsl, ema_activationsl)) / batch_size
                consistency_relation_loss = consistency_weight * consistency_relation_dist * args.consistency_relation_weight
            else:
                consistency_loss = 0.0
                consistency_relation_loss = 0.0
                consistency_weight = 0.0
                consistency_dist = 0.0
             #+ consistency_loss


            z_ = torch.rand((args.generated_batch_size, args.z_dim))
            z_ = z_.cuda()
            G_ = G(z_)

            model.eval()
            _, C_fake_pred = model(G_)

            C_fake_pred = F.softmax(C_fake_pred, dim=1)

            C_fake_wei = torch.max(C_fake_pred, 1)[1]
            C_fake_wei = C_fake_wei.view(-1, 1)
            C_fake_wei = torch.zeros(args.generated_batch_size, 7).cuda().scatter_(1, C_fake_wei, 1)

            C_fake_loss = nll_loss_neg(C_fake_pred, C_fake_wei,wcw2)
            model.train()











            if (epoch > 10) and (args.ema_consistency == 1):
                loss = loss_classification + consistency_loss +  consistency_relation_loss+(generated_weight(epoch) * C_fake_loss)





            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1






















            
            image_batch,  label_batch = image_batch.cuda(), label_batch.cuda()
            # unlabeled_image_batch = ema_image_batch[labeled_bs:]

            # noise1 = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.1, 0.1)
            # noise2 = torch.clamp(torch.randn_like(ema_image_batch) * 0.1, -0.1, 0.1)
             #+ noise2
            inputs = image_batch #+ noise1


            z_ = torch.rand((args.generated_batch_size, args.z_dim))
            z_ = z_.cuda()



            # update D network
            D_optimizer.zero_grad()

            D_real = D(inputs)
            D_real_loss = BCEloss(D_real, torch.ones_like(D_real))

            acul=D.features

            G_ = G(z_)
            D_fake = D(G_)
            acdf=D.features
            D_fake_loss = BCEloss(D_fake, torch.zeros_like(D_fake))




            T_x = transform(G_.detach())
            D_faket = D(T_x)
            acdtf=D.features
            L_D_fake1 = l2loss(D_fake , D_faket )

            consistency_relation_distdf = torch.sum(losses.relation_mse_loss(acdtf, acdf)) / batch_size



            T_x = transform(inputs)
            D_labeledt = D(T_x)

            actul=D.features
            consistency_relation_distul = torch.sum(losses.relation_mse_loss(acul, actul)) / batch_size

            L_reall1 = l2loss(D_real , D_labeledt )

            T_z = z_ + torch.normal(0,0.01,z_.shape).cuda()
            G_T_z = G(T_z)
            D_G_T_z = D(G_T_z.detach())
            actdgtz=D.features
            # zCR: Calculate L_dis |D(G(z)) ? D(G(T(z))|^2
            L_dis = l2loss(D_fake, D_G_T_z)
            consistency_relation_distdtzf = torch.sum(losses.relation_mse_loss(actdgtz, acdf)) / batch_size


            D_loss = D_real_loss + D_fake_loss

            if (epoch>10):
              D_loss = D_real_loss + D_fake_loss+((L_reall1+L_D_fake1)*0.1)+L_dis*0.1+consistency_relation_distul*0.1+consistency_relation_distdf*0.1+consistency_relation_distdtzf*0.1





            D_loss.backward()
            D_optimizer.step()








            # update G network
            G_optimizer.zero_grad()

            G_ = G(z_)

            D_fake = D(G_)
            G_loss_D = BCEloss(D_fake, torch.ones_like(D_fake))
            consistency_relation_distfg = -torch.sum(losses.relation_mse_loss(G_, G_T_z)) / batch_size
            
            L_gen = -l2loss(G_, G_T_z)

            model.eval()
            _, C_fake_pred = model(G_)
            C_fake_pred = F.log_softmax(C_fake_pred, dim=1)
            C_fake_wei = torch.max(C_fake_pred, 1)[1]
            G_loss_C = F.nll_loss(C_fake_pred, C_fake_wei,wcw2)
            model.train()



            G_loss=G_loss_D

            #G_loss = G_loss_D + generated_weight(epoch) * G_loss_C+L_gen*0.5
            if epoch>10:
              G_loss=G_loss_D + generated_weight(epoch) * G_loss_C+L_gen*0.1+consistency_relation_distfg*0.1
            
            G_loss.backward()
            G_optimizer.step()


        with torch.no_grad():
          visualize_results(G, (epoch + 1))
          #time.sleep(300)

        # validate student
        # 

        AUROCs, Accus, Senss, Specs = epochVal_metrics(model, val_dataloader)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()

        logging.info("\nVAL Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nVAL AUROC: {:6f}, VAL Accus: {:6f}, VAL Senss: {:6f}, VAL Specs: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
        
        # test student
        # 
        AUROCs, Accus, Senss, Specs,pre, F1 = epochVal_metrics_test(model, test_dataloader)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        pre_avg = np.array(pre).mean()
        F1_avg = np.array(F1).mean()
        logging.info("\nTEST Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST pre: {:6f}, TEST F1: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg,pre_avg,F1_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))

        # save model
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        torch.save({    'epoch': epoch + 1,
                        'global_step': iter_num,
                        'state_dict': model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'epochs'    : epoch,
                        # 'AUROC'     : AUROC_best,
                   }
                   , save_mode_path
        )
        logging.info("save model to {}".format(save_mode_path))

        # update learning rate
        lr_ = lr_ * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()


