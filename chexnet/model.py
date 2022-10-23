from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pretrained_vit.model import ViT
import json

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os, sys
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E

util_pth = '/mnt/bd/medai-cv/cvpr/CXR8/mae/util'
sys.path.insert(1, util_pth)
import misc
from tqdm import tqdm



use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
device="cuda:1"
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, best_loss, epoch, LR,optimizer, name,out_dir):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model.state_dict(),
        'best_loss': best_loss,
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, out_dir+'checkpoint'+name)


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        loss_scaler):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            debug_dict={}

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            accum_iter=1
            for batch_idx, data in tqdm(enumerate(dataloaders[phase])):
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                loss = criterion(outputs, labels) /accum_iter

                debug_dict[batch_idx] = loss.item()

                # acc_loss += loss
                if phase == 'train':
                    # loss.backward()
                    loss_scaler(loss, optimizer, clip_grad=None,
                            parameters=model.parameters(), create_graph=False,
                            update_grad=(batch_idx + 1) % accum_iter == 0)
                    if (batch_idx + 1) % accum_iter == 0:
                        # optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item() *accum_iter * batch_size


            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))


            jsonString = json.dumps(debug_dict)
            jsonFile = open("debug_original.json", "w")
            jsonFile.write(jsonString)
            jsonFile.close()
            raise ValueError

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results6/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results6/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    loss_scaler = misc.NativeScalerWithGradNormCount()
    # 16

    #16

    try:
        rmtree('results6/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results6/")

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(712),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(712),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    # 224, 448, 640

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    # change densenet to vit
    # model = ViT('B_16_imagenet1k', pretrained=True,image_size=224)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())


    # densenet
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    # add final layer with # outputs in same dimension of labels with sigmoid
    # activation
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    # put model on GPU
    model = model.to(device)
    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,loss_scaler=loss_scaler)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs

def train_cnn_split(PATH_TO_IMAGES, LR, WEIGHT_DECAY,checkpoint_dir=None):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    loss_scaler = misc.NativeScalerWithGradNormCount()
    #16

    # try:
    #     rmtree('results_split2/')
    # except BaseException:
    #     pass  # directory doesn't yet exist, no need to clear it

    try:
        os.makedirs("results_split2/")
    except BaseException:
        pass

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(512),
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    # 224, 448, 640

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset2(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset2(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=2,
        shuffle=True,
        num_workers=1)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    # densenet
    densenet_model = models.densenet121(pretrained=True)
    model = Custom_model(densenet_model,num_classes=14)

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    if checkpoint_dir:
        checkpoint_=torch.load(checkpoint_dir)
        for key in checkpoint_:
            print(key)
        model.load_state_dict(checkpoint_['model'])
        optimizer.load_state_dict(checkpoint_['optimizer'])
        epoch = checkpoint['epoch']
        previous_lr = checkpoint['LR']

    # put model on GPU
    model = model.to(device)

    # train model
    if checkpoint_dir:
        model, best_epoch = train_model2(model, criterion, optimizer, previous_lr, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,loss_scaler=loss_scaler,previous_epoch=epoch)
    else:
        model, best_epoch = train_model2(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,loss_scaler=loss_scaler)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs


class Custom_model(nn.Module):
    def __init__(self,densenet_model,num_classes=14) -> None:
        super().__init__()
        self.shared_feature_extractor = densenet_model.features
        self.classifier = nn.Sequential(nn.Linear(densenet_model.classifier.in_features,num_classes),nn.Sigmoid())

    def forward(self, input,img_dim=448):
        # input is a list of tensors. list len is 4, tensor shape is (batch, 3, 448,448)

        features=[]
        for img_section in input:
            features.append(self.shared_feature_extractor(img_section))

        # combine feature maps. Select element-wise max value at each position
        maximum_feats = features[0]
        for feature in features:
            maximum_feats = torch.maximum(maximum_feats,feature)

        # relu, pooling, classify
        out = F.relu(maximum_feats, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def train_model2(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        loss_scaler,
        previous_epoch=None):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    patience=2
    if previous_epoch:
        start_epoch=previous_epoch
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            accum_iter=2
            for batch_idx, data in tqdm(enumerate(dataloaders[phase])):
                i += 1
                inputs, labels, _ = data
                batch_size = inputs[0].shape[0]
                inputs = [i.to(device) for i in inputs]

                labels = labels.to(device).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                loss = criterion(outputs, labels) /accum_iter

                # acc_loss += loss
                if phase == 'train':
                    loss.backward()
                    if (batch_idx) % accum_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item() *accum_iter * batch_size


            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                if epoch%patience==0:
                    print("decay loss from " + str(LR) + " to " +
                            str(LR / 10) + " as not seeing improvement in val loss")
                    LR = LR / 10
                    # create new optimizer with lower learning rate
                    optimizer = optim.SGD(
                        filter(
                            lambda p: p.requires_grad,
                            model.parameters()),
                        lr=LR,
                        momentum=0.9,
                        weight_decay=weight_decay)
                    print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR,optimizer,str(epoch))

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results_split2/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 10):
            print("no improvement in 5 epochs, break")
            break
    
    checkpoint(model, best_loss, -1, LR,optimizer, 'last')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    # checkpoint_best = torch.load('results_split2/checkpoint')
    # model = checkpoint_best['model']

    return model, best_epoch


class Custom_model2(nn.Module):
    def __init__(self,densenet_model,num_classes=14) -> None:
        super().__init__()
        self.shared_feature_extractor = densenet_model.features
        self.mixer = nn.Sequential(nn.Linear(14*14*4,14*14*4),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(densenet_model.classifier.in_features,num_classes),nn.Sigmoid())

    def forward(self, input,img_dim=448):
        # input is a list of tensors. list len is 4, tensor shape is (batch, 3, 448,448)

        # batch 1024, 14, 14
        all_feature = torch.concat([self.shared_feature_extractor(img_section).unsqueeze(-1) for img_section in input],dim=-1)
        B, C, H, W4 = all_feature.size()
        out = self.mixer(all_feature.reshape(B,C,H*W4))

        # relu, pooling, classify
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out



def train_cnn_split2(PATH_TO_IMAGES, LR, WEIGHT_DECAY,checkpoint_dir=None):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    loss_scaler = misc.NativeScalerWithGradNormCount()

    OUT_DIR="results_split3/"

    try:
        os.makedirs(OUT_DIR)
    except BaseException:
        pass

    # use imagenet mean,std for normalization
    # TODO, calculate mean
    mean = [0.49722]
    std = [0.24979]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    # 224, 448, 640

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset3(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset3(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=4,
        shuffle=True,
        num_workers=1)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    # densenet
    densenet_model = models.densenet121(pretrained=True)
    densenet_model.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Custom_model3(densenet_model,num_classes=14)
    model = model.to(device)

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)

    if checkpoint_dir:
        checkpoint_=torch.load(checkpoint_dir)
        for key in checkpoint_:
            print(key)
        model.load_state_dict(checkpoint_['model'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint_['optimizer'])
        epoch = checkpoint_['epoch']+1
        previous_lr = checkpoint_['LR']
        print(previous_lr)
    else:
        print(LR)

    # train model
    if checkpoint_dir:
        model, best_epoch = train_model3(model, criterion, optimizer, previous_lr, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,loss_scaler=loss_scaler,previous_epoch=epoch,out_dir=OUT_DIR)
    else:
        model, best_epoch = train_model3(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,loss_scaler=loss_scaler,out_dir=OUT_DIR)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES,out_dir=OUT_DIR)

    return preds, aucs


class Custom_model3(nn.Module):
    def __init__(self,densenet_model,num_classes=14) -> None:
        super().__init__()
        self.shared_feature_extractor = densenet_model.features
        self.classifier = nn.Sequential(nn.Linear(densenet_model.classifier.in_features,num_classes),nn.Sigmoid())

    def forward(self, input,img_dim=448):
        # input is (batch, 4, 448, 448)
        features = self.shared_feature_extractor(input) # batch，1024，14，14

        # relu, pooling, classify
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def train_model3(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        loss_scaler,
        previous_epoch=None,
        out_dir="results/"):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    patience=2
    if previous_epoch:
        start_epoch=previous_epoch
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            debug_dict={}

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            accum_iter=1
            for batch_idx, data in tqdm(enumerate(dataloaders[phase])):
                i += 1
                inputs, labels, _ = data
                batch_size = inputs[0].shape[0]
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                outputs = model(inputs)


                # calculate gradient and update parameters in train phase
                loss = criterion(outputs, labels) /accum_iter

                debug_dict[batch_idx]=loss.item()


                # acc_loss += loss
                if phase == 'train':
                    loss.backward()
                    if (batch_idx + 1) % accum_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                # if batch_idx ==10:
                #     break

                running_loss += loss.item()


            # epoch loss is the sum of all losses/ number of iterations
            epoch_loss = running_loss / len(dataloaders[phase])

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # jsonString = json.dumps(debug_dict)
            # jsonFile = open("debug_split.json", "w")
            # jsonFile.write(jsonString)
            # jsonFile.close()
            # raise ValueError

            # decay learning rate if no val loss improvement in this epoch
            if phase == 'val' and epoch_loss <= best_loss:
                improve_epoch=epoch

            if phase == 'val' and epoch_loss > best_loss:
                if epoch - improve_epoch >=patience:
                    print("decay loss from " + str(LR) + " to " +
                            str(LR / 10) + " as not seeing improvement in val loss")
                    LR = LR / 10
                    # create new optimizer with lower learning rate
                    optimizer = optim.SGD(
                        filter(
                            lambda p: p.requires_grad,
                            model.parameters()),
                        lr=LR,
                        momentum=0.9,
                        weight_decay=weight_decay)
                    print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch

            # log training and validation loss over each epoch
            if phase == 'val':
                checkpoint(model, best_loss, epoch, LR,optimizer,str(epoch),out_dir)
                with open(out_dir+"log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 10):
            print("no improvement in 5 epochs, break")
            break
    
    checkpoint(model, best_loss, -1, LR,optimizer, 'last', out_dir)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    # checkpoint_best = torch.load('results_split2/checkpoint')
    # model = checkpoint_best['model']

    return model, best_epoch


class Custom_model4(nn.Module):
    def __init__(self,densenet_model,num_classes=14) -> None:
        super().__init__()
        self.shared_feature_extractor = nn.Sequential(densenet_model.features.conv0,
        densenet_model.features.norm0,
        densenet_model.features.relu0,
        densenet_model.features.pool0,
        densenet_model.features.denseblock1,
        densenet_model.features.transition1,
        densenet_model.features.denseblock2,
        densenet_model.features.transition2,
        densenet_model.features.denseblock3)
        self.classifier = nn.Sequential(nn.Linear(1024,num_classes),nn.Sigmoid())

    def forward(self, input,img_dim=448):
        # input is (batch, 4, 448, 448)
        features = self.shared_feature_extractor(input) 

        # relu, pooling, classify
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out



def train_cnn_split3(PATH_TO_IMAGES, LR, WEIGHT_DECAY,checkpoint_dir=None):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    loss_scaler = misc.NativeScalerWithGradNormCount()
    #16

    OUT_DIR="results_split5/"

    try:
        os.makedirs(OUT_DIR)
    except BaseException:
        pass

    mean = [0.49722]
    std = [0.24979]

    N_LABELS = 14  # we are predicting 14 labels

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    # 224, 448, 640

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset3(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset3(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    # densenet
    densenet_model = models.densenet121(pretrained=True)
    densenet_model.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = Custom_model4(densenet_model,num_classes=14)

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    if checkpoint_dir:
        checkpoint_=torch.load(checkpoint_dir)
        for key in checkpoint_:
            print(key)
        model.load_state_dict(checkpoint_['model'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint_['optimizer'])
        epoch = checkpoint_['epoch']+1
        previous_lr = checkpoint_['LR']


    # train model
    if checkpoint_dir:
        model, best_epoch = train_model3(model, criterion, optimizer, previous_lr, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,loss_scaler=loss_scaler,previous_epoch=epoch,out_dir=OUT_DIR)
    else:
        model, best_epoch = train_model3(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,loss_scaler=loss_scaler,out_dir=OUT_DIR)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES,out_dir=OUT_DIR)

    return preds, aucs