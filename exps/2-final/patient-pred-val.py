from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import random

BATCH_SIZE = 16
EPOCHS = 50
SEED = 13
LR = 1e-3

NN_LAYERS = [16, 12, 8]

TRAINFT = "train_proba.ft"
VALFT = "val_proba.ft"

set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#calculate kappa, F-1 socre and AUC value
def odir_metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.numpy().flatten()
    pr = pr_data.numpy().flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0

    return final_score

model = Sequential()
#model.add(Dropout(0.2, input_shape=(NN_LAYERS[0],)))
model.add(Dense(NN_LAYERS[1], activation='relu', 
kernel_initializer='he_normal', input_shape=(NN_LAYERS[0],)))
#model.add(Dropout(0.2))
model.add(Dense(NN_LAYERS[2], activation='sigmoid'))

model.compile(optimizer=Adam(lr=LR),
              loss='binary_crossentropy',
              metrics=odir_metrics,
              run_eagerly=True)

df_val = pd.read_feather(VALFT)

val_eyeprob = df_val.iloc[:,1].to_numpy()
val_eyeprob = np.array([list(i) for i in val_eyeprob])

val_ytrue = df_val.iloc[:,2].to_numpy()
val_ytrue = np.array([list(i) for i in val_ytrue])

model.load_weights("pacient_pred_baseline_proba_best_model.h5")

print(model.evaluate(val_eyeprob, val_ytrue))