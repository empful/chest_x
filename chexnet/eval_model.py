import torch
import pandas as pd
import cxr_dataset as CXR
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
device="cuda:1"


def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES,out_dir):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader
    dataset = CXR.CXRDataset3(
        path_to_images=PATH_TO_IMAGES,
        fold="test", #test
        transform=data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in tqdm(enumerate(dataloader)):

        inputs, labels, _ = data
        labels = Variable(labels.to(device))

        # inputs =[i.to(device) for i in inputs]
        inputs = inputs.to(device)
        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        # if(i % 10 == 0):
        #     print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if column not in [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
                'Hernia']:
                    continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                np.array(actual), np.array(pred))
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    pred_df.to_csv(out_dir+"preds.csv", index=False)
    auc_df.to_csv(out_dir+"aucs.csv", index=False)
    return pred_df, auc_df


class Custom_model3(nn.Module):
    def __init__(self,densenet_model,num_classes=14) -> None:
        super().__init__()
        self.shared_feature_extractor = densenet_model.features
        self.classifier = nn.Sequential(nn.Linear(densenet_model.classifier.in_features,num_classes),nn.Sigmoid())

    def forward(self, input,img_dim=448):
        # input is (batch, 4, 448, 448)
        features = self.shared_feature_extractor(input) # 8，1024，14，14

        # relu, pooling, classify
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# code for evaluate checkpoint:


# densenet_model = models.densenet121(pretrained=False)
# densenet_model.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model = Custom_model3(densenet_model,num_classes=14)
#
#
# out_dir="results_512/"
# checkpoint_best = torch.load("results_512/checkpoint")
# model.load_state_dict(checkpoint_best['model'])
# model.to(device)
# # use imagenet mean,std for normalization
# # mean = [0.485, 0.456, 0.406]
# # std = [0.229, 0.224, 0.225]
#
# mean = [0.49722]
# std = [0.24979]
#
# data_transforms = {
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ]),
#     }
# PATH_TO_IMAGES = "/mnt/bd/medai-cv/cvpr/CXR8/images/images"
# preds, aucs = make_pred_multilabel(
#         data_transforms, model, PATH_TO_IMAGES, out_dir=out_dir)
#
#
#
# aucs = pd.read_csv(out_dir +"aucs.csv")
# print("average aucs", np.mean(aucs.auc))
# print(aucs)