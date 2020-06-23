# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import PCB
import json
from shutil import copyfile
import pkg_resources

import multiprocessing

#os.environ["PYTORCH_JIT"] = "0"

def get_net(is_parallel, net):
    return net.module if is_parallel else net

######################################################################
# Training the model
# ------------------
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, log_file, stage, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    last_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # Due to the GPU memory limitations, we don't use the 'val' dataset
        for phase in ['train', 'val']:#, 'val'
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                part = {}
                sm = nn.Softmax(dim=1)
                num_part = 6

                for i in range(num_part):
                    part[i] = outputs[i]

                score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
                _, preds = torch.max(score.data, 1)

                loss = criterion(part[0], labels)
                for i in range(num_part - 1):
                    loss += criterion(part[i + 1], labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            log_file.write('{} epoch : {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase,
                                                                           epoch_loss, epoch_acc) + '\n')

            if phase == 'val':
            #if phase == 'train':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch, stage)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    save_network(model, 'best', stage)
    model.load_state_dict(last_model_wts)
    save_network(model, 'last', stage)
    return model

######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label, stage):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(save_dir, stage)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if is_parallel_train:
        torch.save(network.module.state_dict(), save_path + '/' + save_filename)
    else:
        torch.save(network.cpu().state_dict(), save_path + '/' + save_filename)
        if torch.cuda.is_available():
            network.cuda(gpu_ids[0])

######################################################################
# load model
# ---------------------------
def load_network(network, stage):
    save_path = os.path.join(args.save_dir, stage, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# PCB train
# ------------------
# Step1 : train the PCB model
# According to original paper, we set the difference learning rate for difference layers.

def pcb_train(model, criterion, log_file, stage, num_epoch):
    ignored_params = list(map(id, get_net(is_parallel_train, model).classifiers.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, get_net(is_parallel_train, model).parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1*(args.PCB_LearnRate)},
        {'params': get_net(is_parallel_train, model).classifiers.parameters(), 'lr': args.PCB_LearnRate},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        log_file, stage, num_epochs=num_epoch)
    return model

######################################################################
# RPP train
# ------------------
# Setp 2&3: train the rpp layers
# According to original paper, we set the learning rate at 0.01 for rpp layers.

def rpp_train(model, criterion, log_file, stage, num_epoch):

    # ignored_params = list(map(id, get_net(opt, model).avgpool.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, get_net(opt, model).parameters())
    # optimizer_ft = optim.SGD([
    #     {'params': base_params, 'lr': 0.00},
    #     {'params': get_net(opt, model).avgpool.parameters(), 'lr': 0.01},
    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer_ft = optim.SGD(get_net(is_parallel_train, model).avgpool.parameters(), lr=args.RPP_LearnRate,
                              weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 100 epochs (never use)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        log_file, stage, num_epochs=num_epoch)
    return model

######################################################################
# full train
# ------------------
# Step 4: train the whole net
# According to original paper, we set the difference learning rate for the whole net

def full_train(model, criterion, log_file, stage, num_epoch):
    ignored_params = list(map(id, get_net(is_parallel_train, model).classifiers.parameters()))
    ignored_params += list(map(id, get_net(is_parallel_train, model).avgpool.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, get_net(is_parallel_train, model).parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1*(args.full_LearnRate)},
        {'params': get_net(is_parallel_train, model).classifiers.parameters(), 'lr': args.full_LearnRate},
        {'params': get_net(is_parallel_train, model).avgpool.parameters(), 'lr': args.full_LearnRate},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 100 epochs (never use)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        log_file, stage, num_epochs=num_epoch)
    return model

######################################################################
# Start training
# ---------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()

    version = torch.__version__
    ######################################################################
    # Options #
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')

    if os.name == 'nt':  #win
        parser.add_argument('--save_dir', default='./model/',
                            type=str, help='save model dir')
        parser.add_argument('--data_dir', default='../data/Market/pytorch', type=str, # D:/workspace/ML/BPM/project
                        help='training dir path')
    elif os.name == 'posix':  #linux
        parser.add_argument('--save_dir', default='./model/',
                            type=str, help='save model dir')
        parser.add_argument('--data_dir', default='../data/Market/pytorch', type=str,
                        help='training dir path')
    # parser.add_argument('--data_dir', default='../data/cuhk03_release/pytorch', type=str, help='training dir path')
    # parser.add_argument('--data_dir', default='../data/Market/pytorch', type=str, help='training dir path')

    parser.add_argument('--train_all', action='store_true', default=True, help='use all training data')
    parser.add_argument('--PCB_epoch', default=60, type=int, help='PCB epoch')  # 60
    parser.add_argument('--RPP_epoch', default=5, type=int, help='RPP epoch')
    parser.add_argument('--full_epoch', default=10, type=int, help='full epoch')
    parser.add_argument('--PCB_LearnRate', default=0.1, type=float, help='PCB learning rate')
    parser.add_argument('--RPP_LearnRate', default=0.01, type=float, help='RPP learning rate')
    parser.add_argument('--full_LearnRate', default=0.01, type=float, help='full learning rate')
    parser.add_argument('--batchsize', default=32, type=int, help='batch_size')  # 32
    parser.add_argument('--RPP', action='store_true', help='use RPP', default=True)
    parser.add_argument('--jump_to_RPP', default=False, action='store_true', help='ignore the PCB train')
    args = parser.parse_args()

    # setGPU #
    use_gpu = torch.cuda.is_available()
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    is_parallel_train = False
    if len(gpu_ids) > 1:
        is_parallel_train = True

    # setSeed #
    seed = 1994
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ------ #
    data_dir = args.data_dir
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    ######################################################################
    # Load Data
    # ---------
    #
    transform_train_list = [
        transforms.Resize((384, 128), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_val_list = [
        transforms.Resize(size=(384, 128), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    train_all = ''
    if args.train_all:
        train_all = '_all'

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                                   data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                                 data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchsize,
                                                  shuffle=True, num_workers=8)  # 8 workers may work faster
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Record the training information #
    f = open(args.save_dir + 'train_log.txt', 'w')

    copyfile('./train.py', args.save_dir + '/train.py')
    copyfile('./model.py', args.save_dir + '/model.py')

    # step1: PCB training #
    stage = 'PCB'
    model = PCB(len(class_names))
    if use_gpu:
        model = model.cuda()
    if is_parallel_train:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    criterion = nn.CrossEntropyLoss()

    if args.jump_to_RPP==True:
        trained_model = PCB(len(class_names))
        trained_model = load_network(trained_model, 'PCB')
        model = trained_model
    else:
        print('-------PCB train-----------')
        model = pcb_train(model, criterion, f, stage, args.PCB_epoch) #60

    ############################
    # step2&3: RPP training #
    if args.RPP:
        stage = 'RPP'
        if args.jump_to_RPP:
            model = model.convert_to_rpp()
        else:
            model = get_net(is_parallel_train, model).convert_to_rpp()

        if use_gpu:
            model = model.cuda()
        if is_parallel_train:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        print('-------RPP train-----------')
        model = rpp_train(model, criterion, f, stage, args.RPP_epoch)

        ############################
        # step4: whole net training #
        stage = 'full'

        print('-------full train-----------')
        full_train(model, criterion, f, stage, args.full_epoch)
    f.close()

    #input('please press enter')





