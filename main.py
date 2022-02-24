from __future__ import absolute_import,print_function
from operator import mod
import os
import sys
import time
import datetime
import argparse
import collections
import random
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import scipy.io as sio
import warnings

from sklearn.cluster import DBSCAN

from configs.default import get_config
from data import build_dataloader
import data
from models import build_model
from losses import build_losses
from tools.eval_metrics import evaluate,extract_features
from tools.utils import AverageMeter, Logger, save_checkpoint, set_seed
from tools.trainers import DualClusterContrastTrainer
from data import transforms as T
from data.Preprocessor import Preprocessor
from data.samplers import RandomMultipleGallerySampler

from IPython import embed


def parse_option():
    parser = argparse.ArgumentParser(description='Train image-based re-id model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, help="market1501, cuhk03, dukemtmcreid, msmt17")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    if not config.EVAL_MODE:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_test.txt'))
    print("==========\nConfig:{}\n==========".format(config))
    print("Currently using GPU {}".format(config.GPU))
    # Set random seed
    set_seed(config.SEED)

    # Build dataloader
    trainloader, queryloader, galleryloader, initloader, num_classes = build_dataloader(config)
    #print("num_classes={}\n".format(num_classes))
    # Build model
    model, classifier = build_model(config, num_classes)
    # Build classification and pairwise loss
    criterion_cla, criterion_pair = build_losses(config)
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    model = nn.DataParallel(model).cuda()
    classifier = nn.DataParallel(classifier).cuda()

    if config.EVAL_MODE:
        print("Evaluate only")
        test(model, queryloader, galleryloader)
        return

    #new
    print("==> Load unlabeled dataset")
    dataset = get_data(config.DATA.DATASET, config.DATA.ROOT)
    test_loader = get_test_loader(dataset, config.DATA.HEIGHT, config.DATA.WIDTH, config.DATA.TRAIN_BATCH, config.DATA.NUM_WORKERS)

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    trainer=DualClusterContrastTrainer(model)

    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        #new
        if epoch==0:
            with torch.no_grad():
                cluster_loader = get_test_loader(dataset, config.DATA.HEIGHT, config.DATA.WIDTH, config.DATA.TRAIN_BATCH, config.DATA.NUM_WORKERS, testset=sorted(dataset.train)) 
                features, labels = extract_features(model, cluster_loader, print_freq=50)
                features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                labels = torch.cat([labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers
            pseudo_labels = labels.data.cpu().numpy()
    
            cluster_features = generate_cluster_features(pseudo_labels, features)

            del cluster_loader, features

            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            print("cluster_features size={}\n".format(cluster_features.size()))
            print("cluster_features={}\n".format(cluster_features))

            dcc_loss = DCCLoss(2048,num_cluster,weight= config.w, momentum = config.momentum, init_feat=F.normalize(cluster_features, dim=1).cuda())
            trainer.loss = dcc_loss

        train_loader = get_train_loader(config, dataset, config.DATA.HEIGHT, config.DATA.WIDTH, config.DATA.TRAIN_BATCH, config.DATA.NUM_WORKERS, config.DATA.NUM_INSTANCES)
        cluster_features = trainer.train(epoch, train_loader, optimizer,print_freq=config.print_freq)


        #start_train_time = time.time()
        #train(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader)
        #train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            state_dict = model.module.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    
    for batch_idx, (imgs, pids, indexes) in enumerate(trainloader):
        imgs, pids,indexes= imgs.cuda(), pids.cuda(),indexes.cuda()

        #print("pids={}\n".format(pids))
        #print("batch_idx={}/n".format(batch_idx))
        print("indexes={}\n".format(indexes))

        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(imgs) #shape 64*2048,data such as 0.xxxx,1.xxxx,-0.xxxx,-1.xxxx
        outputs = classifier(features) #也是小数，但是比features小得多 64*751
        
        _, preds = torch.max(outputs.data, 1) #preds是每一张image从属于哪一类的预测
        #print("preds={}/n".format(preds))

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        #print("cla_loss={}\n".format(cla_loss))
        pair_loss = criterion_pair(features, pids)
        loss = cla_loss + pair_loss     
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'ClaLoss:{cla_loss.avg:.4f} '
          'PairLoss:{pair_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time, data_time=data_time, 
           cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, acc=corrects))


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)

    return img_flip


@torch.no_grad()
def extract_feature(model, dataloader):
    features, pids, camids = [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids) in enumerate(dataloader):
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs).data.cpu()
        batch_features_flip = model(flip_imgs).data.cpu()
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()

    return features, pids, camids


def test(model, queryloader, galleryloader):
    since = time.time()
    model.eval()
    # Extract features for query set
    qf, q_pids, q_camids = extract_feature(model, queryloader)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # Extract features for gallery set
    gf, g_pids, g_camids = extract_feature(model, galleryloader)
    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    if config.TEST.DISTANCE == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        # Cosine similarity
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------------------------------------")

    return cmc[0]


#new
def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print("root={}".format(root))
    dataset = data.create(name, root)
    
    return dataset

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    #print("dataset.attr={}\n".format(getattr(dataset)))
    test_loader = DataLoader(
        Preprocessor(testset, root=None, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances,  trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = DataLoader(Preprocessor(train_set, root=None, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True)

    return train_loader

from torch import nn,autograd
import random

class DCC(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut_ccc, lut_icc,  momentum):
        ctx.lut_ccc = lut_ccc
        ctx.lut_icc = lut_icc
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs_ccc = inputs.mm(ctx.lut_ccc.t())
        outputs_icc = inputs.mm(ctx.lut_icc.t())

        return outputs_ccc,outputs_icc

    @staticmethod
    def backward(ctx, grad_outputs_ccc, grad_outputs_icc):
        inputs,targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs_ccc.mm(ctx.lut_ccc)+grad_outputs_icc.mm(ctx.lut_icc)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.data.cpu().numpy()):
            batch_centers[index].append(instance_feature)

        for y, features in batch_centers.items():
            mean_feature = torch.stack(batch_centers[y],dim=0)
            non_mean_feature = mean_feature.mean(0)
            x = F.normalize(non_mean_feature,dim=0)
            ctx.lut_ccc[y] = ctx.momentum * ctx.lut_ccc[y] + (1.-ctx.momentum) * x
            ctx.lut_ccc[y] /= ctx.lut_ccc[y].norm()

        del batch_centers 

        for x, y in zip(inputs,targets.data.cpu().numpy()):
            ctx.lut_icc[y] = ctx.lut_icc[y] * ctx.momentum + (1 - ctx.momentum) * x
            ctx.lut_icc[y] /= ctx.lut_icc[y].norm()

        return grad_inputs, None, None, None, None


def oim(inputs, targets, lut_ccc, lut_icc, momentum=0.1):
    return DCC.apply(inputs, targets, lut_ccc, lut_icc, torch.Tensor([momentum]).to(inputs.device))

import copy
class DCCLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.0,
                 weight=None, size_average=True,init_feat=[]):
        super(DCCLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        #print("num_classes={}/n".format(self.num_classes))
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        #print("self.size_average size={}\n".format(self.size_average.size()))
        print("self.size_average={}\n".format(init_feat))

        self.register_buffer('lut_ccc', torch.zeros(num_classes, num_features).cuda())
        self.lut_ccc = copy.deepcopy(init_feat)

        self.register_buffer('lut_icc', torch.zeros(num_classes, num_features).cuda())
        self.lut_icc = copy.deepcopy(init_feat)
        print('Weight:{},Momentum:{}'.format(self.weight,self.momentum))

    def forward(self, inputs,  targets):
        inputs_ccc,inputs_icc = oim(inputs, targets, self.lut_ccc, self.lut_icc, momentum=self.momentum)

        inputs_ccc *= self.scalar
        inputs_icc *= self.scalar

        loss_ccc = F.cross_entropy(inputs_ccc, targets, size_average=self.size_average)
        loss_icc = F.cross_entropy(inputs_icc, targets, size_average=self.size_average)



        loss_con = F.smooth_l1_loss(inputs_ccc, inputs_icc.detach(), reduction='elementwise_mean')
        loss = loss_ccc+loss_icc+self.weight*loss_con

        return loss


if __name__ == '__main__':
    config = parse_option()
    main(config)