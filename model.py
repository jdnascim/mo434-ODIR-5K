#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip3 install pytorch_lightning
# !pip3 install opencv-python
# !pip3 install scikit-learn


# In[2]:


import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Datasets import Modes, ODIR5K

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import cv2

import torchvision.transforms as transforms
from torchvision import models

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[3]:


BS = 32
INPUT_IMG = 299
debug=True
LR = 5e-5
MAX_EPOCHS = 50
TRAIN_SIZE = 0.8


# In[4]:


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class SwavFinetuning(pl.LightningModule):
    def __init__(self, freeze=False, classes=2):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/swav', 'resnet50')
        
        self.d_dim = self.model.fc.in_features
        
        self.model.fc = Identity()
        
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        
        self.classes = classes
        self.linear_clf = nn.Linear(self.d_dim, self.classes)

    def forward(self, images):        
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        representations = self(images)
        logits = self.linear_clf(representations)
        
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        return loss
        
    def to_device(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        return x, y
    
    
    def shared_step(self, batch, step):
        images, labels = batch
        representations = self(images).detach()
        
        logits = self.linear_clf(representations)
        loss = F.cross_entropy(logits, labels)
        
        self.log(f'{step}_loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        return {"probs": F.softmax(logits, dim=1),
                "labels": labels
            
        }
    
#     def shared_end(self, outputs, step):
#         prob_preds = []
#         labels = []
        
#         for output in outputs:
#             probs, y = output["probs"], output["labels"] 
#             prob_preds.append(probs.cpu().numpy())
#             labels.extend(y.cpu().numpy())
        
#         prob_preds = np.concatenate(prob_preds)
        
#         labels_onehot = preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(labels).reshape(len(labels), 1))
        
#         final_score = self.odir_metric(labels_onehot, prob_preds)
#         self.log(f"odir_score_{step}", final_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
#         auc = roc_auc_score(labels, prob_preds)
#         self.log(f"auc_score_{step}", auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def shared_end(self, outputs, loader, step):
        eval_loader = loader
        
        prob_preds = []
        labels = []
        y_pred = []
        for batch in eval_loader:
            with torch.no_grad():
                x, y = self.to_device(batch, self.device)
                representations = self.model(x).detach()
                mlp_preds = self.linear_clf(representations)
                probs = F.softmax(mlp_preds, dim=1)
                prob_preds.append(probs.cpu().numpy())
                labels.extend(y.cpu().numpy())
                y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
        
        
        prob_preds = np.concatenate(prob_preds)
        auc = metrics.roc_auc_score(labels, prob_preds, average='weighted', multi_class='ovo')
        self.log(f"auc_score_{step}", auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        labels_onehot = preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(labels).reshape(len(labels), 1))
        final_score = self.odir_metric(labels_onehot, prob_preds)
        self.log(f"odir_score_{step}", final_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if step == 'test':
            cm = metrics.confusion_matrix(labels, y_pred)
            target_names = ['N', 'D', 'G', 'C', "A", "H", "M", "O"]
            plot_confusion_matrix(cm, target_names=target_names, auc=auc, normalize=False)
            print(cm)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')
    
    def validation_epoch_end(self, outputs):
        self.shared_end(outputs, self.val_dataloader(), 'val')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')
    
    def test_epoch_end(self, outputs):
        self.shared_end(outputs, self.test_dataloader(), 'test')
            
    def odir_metric(self, gt_data, pr_data):
        th = 0.5
        gt = gt_data.flatten()
        pr = pr_data.flatten()
        kappa = metrics.cohen_kappa_score(gt, pr>th)
        f1 = metrics.f1_score(gt, pr>th, average='micro')
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa+f1+auc)/3.0
        
        return final_score
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def plot_confusion_matrix(cm,
                          target_names,
                          auc,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    
    code taken from: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label\nAUC={:0.4f};misclass={:0.4f}'.format(auc, misclass))
    plt.savefig("cm_best", bbox_inches='tight', pad_inches=0)
    plt.tight_layout()
    plt.show()


def plot_cm(y_true, y_pred, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

# In[5]:


# logger = None
# debug = False

# color_transform = [get_color_distortion(), RandomGaussianBlur()]
# mean = [0.485, 0.456, 0.406]
# std = [0.228, 0.224, 0.225]
# trans = []
# randomresizedcrop = transforms.RandomResizedCrop(INPUT_IMG)
# trans = transforms.Compose([
#     randomresizedcrop,
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.Compose(color_transform),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)])

# train_t = trans

# #init the dataset withou any augmentation
# full_train = ODIR5K(Modes.train, None)

# #get the train size
# train_size = int(TRAIN_SIZE * len(full_train))
# #calculate the validation size
# val_size = len(full_train) - train_size

# #split the datasts
# odir_train, odir_val = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))
# #trick to disantangle the agumentations variable from train to validation
# odir_train.dataset = copy(full_train)

# #set the train augmentations
# odir_train.dataset.aug = train_t

# #build the validation augmentations
# trans = []
# randomresizedcrop = transforms.RandomResizedCrop(INPUT_IMG)
# trans = transforms.Compose([
#     randomresizedcrop,
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)])

# test_t = trans
# odir_val.dataset.aug = test_t

# odir_test = ODIR5K(Modes.test, test_t)
# logger = CSVLogger("/odir5k/exps/2-final/pl-logs", name="odir5k-swav-ft")
# dl_train = DataLoader(odir_train, batch_size=BS, shuffle=True, num_workers=4)
# dl_val = DataLoader(odir_val, batch_size=BS, shuffle=False, num_workers=4)
# dl_test = DataLoader(odir_test, batch_size=BS, shuffle=False, num_workers=4)

# checkpoint = ModelCheckpoint(monitor='odir_score_val',
#                              dirpath='/odir5k/exps/2-final/',
#                              filename='swav-r50-{epoch}-{odir_score_val:.3f}-{val_loss:.3f}-{auc_score_val:.3f}',
#                             mode='max')

# early_stopping = EarlyStopping(monitor='val_loss', 
#                                patience=10, 
#                                mode='min')

# model = SwavFinetuning(classes=8) 

# trainer = pl.Trainer(gpus=[2],
#                      logger=logger,
#                      callbacks=[early_stopping, checkpoint],
#                      fast_dev_run=debug,
#                      max_epochs=MAX_EPOCHS)

# # trainer = pl.Trainer(gpus=[1],
# #                      logger=logger,
# #                      callbacks=[early_stopping],
# #                      fast_dev_run=debug,
# #                      max_epochs=15
# #                     )

# trainer.fit(model, dl_train, dl_val)
# trainer.test(model, dl_test)


# In[ ]:




