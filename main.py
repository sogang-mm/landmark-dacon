import pandas as pd
import numpy as np

import argparse
import torch
from torchvision import transforms as trn
from torch.utils.data import DataLoader

from torch import nn, optim
from tqdm import tqdm
from utils import AverageMeter, gap

import os
import logging
from datetime import datetime
from tensorboardX import SummaryWriter
from autoaugment import ImageNetPolicy

from dataset import TrainDataset, TestDataset
from models import Resnet50


def init_logger(save_dir):
    c_time = datetime.now().strftime("%Y%m%d/%H%M%S")
    log_dir = os.path.join(save_dir, c_time)
    log_txt = os.path.join(log_dir, 'log.txt')

    global writer
    writer = SummaryWriter(log_dir)
    global logger
    logger = logging.getLogger(c_time)

    logger.setLevel(logging.INFO)
    logger = logging.getLogger(c_time)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    h_file = logging.FileHandler(filename=log_txt, mode='a')
    h_file.setFormatter(fmt)
    h_file.setLevel(logging.INFO)
    logger.addHandler(h_file)
    logger.info(f'Log directory ... {log_txt}')
    return log_dir


def train(model, loader, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    pbar = tqdm(loader, ncols=150)
    y_true = dict()
    y_pred = dict()

    softmax = nn.Softmax(dim=1)

    for i, (image, iid, target, _) in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(image.cuda())
        outputs = softmax(outputs)
        loss = criterion(outputs, target.cuda())
        loss.backward()
        optimizer.step()

        conf, indice = torch.topk(outputs, k=5)
        indice = indice.cpu()

        y_true.update({k: t for k, t in zip(iid, target.numpy())})
        y_pred.update({k: (t, c) for k, t, c in
                       zip(iid, indice[:, 0].cpu().detach().numpy(), conf[:, 0].cpu().detach().numpy())})

        top1.update(torch.sum(indice[:, :1] == target.view(-1, 1)).item())
        top5.update(torch.sum(indice == target.view(-1, 1)).item())
        losses.update(loss)

        log = f'[Epoch {epoch}] '
        log += f'Train loss : {losses.val:.4f}({losses.avg:.4f}) '
        log += f'Top1 : {top1.val / loader.batch_size:.4f}({top1.sum / (i * loader.batch_size):.4f}) '
        log += f'Top5 : {top5.val / loader.batch_size:.4f}({top5.sum / (i * loader.batch_size):.4f})'
        pbar.set_description(log)
        pbar.update()

    _lr = optimizer.param_groups[0]['lr']
    _gap = gap(y_true, y_pred)
    log = f'[EPOCH {epoch}] Train Loss : {losses.avg:.4f}, '
    log += f'Top1 : {top1.sum / loader.dataset.__len__():.4f}, '
    log += f'Top5 : {top5.sum / loader.dataset.__len__():.4f}, '
    log += f'GAP : {_gap:.4e}, '
    log += f'LR : {_lr:.2e}'

    logger.info(log)
    pbar.set_description(log)
    pbar.close()

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/top1', top1.sum / loader.dataset.__len__(), epoch)
    writer.add_scalar('train/top5', top5.sum / loader.dataset.__len__(), epoch)
    writer.add_scalar('train/GAP', _gap, epoch)
    writer.add_scalar('train/LR', _lr, epoch)


@torch.no_grad()
def test(model, loader, epoch, log_dir):
    model.eval()
    pbar = tqdm(loader, ncols=150)
    iids = []
    classes = []
    confideneces = []
    softmax = nn.Softmax(dim=1)
    for i, (image, iid) in enumerate(loader, start=1):
        outputs = model(image.cuda())
        outputs = softmax(outputs)
        conf, indice = torch.topk(outputs, k=1)
        iids.extend(iid)
        classes.extend(indice[:, 0].cpu().numpy())
        confideneces.extend(conf[:, 0].cpu().numpy())
        pbar.update()

    pbar.close()
    iids = pd.Series(iids, name="id")
    classes = pd.Series(classes, name="landmark_id")
    confideneces = pd.Series(confideneces, name="conf")

    df = pd.concat([iids, classes, confideneces], axis=1)
    df.to_csv(f'{log_dir}/submission_ep{epoch:03d}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category_csv', dest='category_csv',
                        default="/mldisk/nfs_shared_/ms/landmark_dacon/public/category.csv")

    parser.add_argument('--train_dir', dest='train_dir',
                        default="/mldisk/nfs_shared_/ms/landmark_dacon/public/train_lnk/")
    parser.add_argument('--train_csv', dest='train_csv',
                        default="/mldisk/nfs_shared_/ms/landmark_dacon/public/train.csv")

    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default="/hdd/ms/landmark_dacon_ckpt/")

    parser.add_argument('--test_dir', dest='test_dir',
                        default="/mldisk/nfs_shared_/ms/landmark_dacon/public/test/")
    parser.add_argument('--submission_csv', dest='submission_csv',
                        default="/mldisk/nfs_shared_/ms/landmark_dacon/public/sample_submission.csv")

    ##
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)

    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0)

    parser.add_argument('-step', '--step_size', type=int, default=10)
    parser.add_argument('-gamma', '--step_gamma', type=float, default=0.8)

    args = parser.parse_args()

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    train_trn = trn.Compose([
        # trn.RandomResizedCrop(224),
        # trn.RandomHorizontalFlip(),
        # ImageNetPolicy(),
        trn.Resize((256, 256)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_trn = trn.Compose([
        trn.Resize((256, 256)),
        trn.ToTensor()])

    category = [i[1] for i in pd.read_csv(args.category_csv).values.tolist()]

    train_dataset = TrainDataset(args.train_dir, args.train_csv, category, train_trn)
    test_dataset = TestDataset(args.test_dir, args.submission_csv, category, test_trn)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    log_dir = init_logger(args.ckpt_dir)
    logger.info(args)

    model = Resnet50().cuda()
    for n, p in model.named_parameters():
        p.requires_grad = True if n.startswith('fc') else False
    model = nn.DataParallel(model.cuda())


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.step_gamma)

    for ep in range(1, args.epochs):
        train(model, train_loader, criterion, optimizer, ep)
        test(model, test_loader, ep, log_dir)
        scheduler.step()

        torch.save({'model_state_dict': model.module.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': ep, },
                   f'{log_dir}/ckpt_epoch_{ep:03d}.pt')
