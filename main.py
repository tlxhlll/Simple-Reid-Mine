from __future__ import absolute_import,print_function
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
from data import transforms as T
from data.Preprocessor import Preprocessor

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
    dataset = get_data(config.data.dataset, config.data.data_dir)
    test_loader = get_test_loader(dataset, config.height, config.width, config.batch_size, config.workers)

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")



    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        #new
        if epoch==0:
            with torch.no_grad():
                cluster_loader = get_test_loader(dataset, config.height, config.width,config.batch_size, config.workers, testset=sorted(dataset.train)) 
                features, labels = extract_features(model, cluster_loader, print_freq=50)
                features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                labels = torch.cat([labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                print("feature cat size={}/n".format(features.size()))
        
        start_train_time = time.time()
        train(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader)
        train_time += round(time.time() - start_train_time)        
        
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
    
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        imgs, pids = imgs.cuda(), pids.cuda()

        #print("pids={}\n".format(pids))
        #print("batch_idx={}/n".format(batch_idx))

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
    print(root)
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

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

if __name__ == '__main__':
    config = parse_option()
    main(config)