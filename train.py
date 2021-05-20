# -*- coding=utf-8 -*-
'''
# @filename  : train.py
# @author    : cjr
# @date      : 2021-4-15
# @brief     : train shuffle net
'''
import argparse

from shufflenet import shufflenet
from shufflenetv2 import shufflenetv2
from shufflev2se import shufflenetv2se
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.autograd import Variable
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR

import time

CIFAR10_TRAIN_MEAN = (0.485, 0.456, 0.406)
CIFAR10_TRAIN_STD = (0.229, 0.224, 0.225)

# total training epoches
EPOCH = 200
MILESTONES = [30, 60, 100]


def train(epoch):
    net.train()
    for batch_index, (images, labels) in enumerate(cifar10_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images, labels = images.to(device), labels.to(device)

        labels = labels
        images = images

        optimizer.zero_grad()
        outputs = net(images)
        correct = 0
        pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = loss_function(outputs, labels)
        accuracy = (100. * correct) / len(outputs)

        loss.backward()
        optimizer.step()
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAccuracy: {:0.6f}'.format(
            loss.item(),
            accuracy,
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar10_training_loader.dataset)
        ))


def eval_training(epoch):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    with torch.no_grad():
        for (images, labels) in cifar10_test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar10_test_loader.dataset),
        correct.float() / len(cifar10_test_loader.dataset)
    ))
    print()
    return correct.float() / len(cifar10_test_loader.dataset)


def xavier(param):
    torch.nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        # m.bias.data.zero_()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = shufflenetv2se(1.)
    net.conv1.apply(weights_init)
    net.maxpool.apply(weights_init)
    net.stage2.apply(weights_init)
    net.stage3.apply(weights_init)
    net.stage4.apply(weights_init)
    net.conv_last.apply(weights_init)
    model = net.to(device)

    # net = shufflenet()
    # load = torch.load("shuffle_default.pth")
    # net.load_state_dict(load)
    # net.conv_head.apply(weights_init)
    # net.max_pool.apply(weights_init)
    # net.stage2.apply(weights_init)
    # net.stage3.apply(weights_init)
    # net.stage4.apply(weights_init)
    # net = net.to(device)

    # data preprocessing:
    cifar10_training_loader = get_training_dataloader(
        CIFAR10_TRAIN_MEAN,
        CIFAR10_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    cifar10_test_loader = get_test_dataloader(
        CIFAR10_TRAIN_MEAN,
        CIFAR10_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar10_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    best_acc = 0.0
    a = time.time()
    for epoch in range(1, EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), "shuffle_default.pth")
    b = time.time()
    print("total time cost:{:.2f}".format(b - a))
    print("best acc:{:.3f}".format(best_acc))

    # shuffle v2
    # total
    # time
    # cost: 11837.58
    # best
    # acc: 0.731
