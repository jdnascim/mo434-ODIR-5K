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

NN_LAYERS = [3072, 50, 8]

TRAINFT = "train_features.ft"
VALFT = "val_features.ft"

EXP_NAME = "pacient_pred_baseline_features"

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
model.add(Dense(NN_LAYERS[1], activation='relu', 
kernel_initializer='he_normal', input_shape=(NN_LAYERS[0],)))
model.add(Dense(NN_LAYERS[2], activation='sigmoid'))

model.compile(optimizer=Adam(lr=LR),
              loss='binary_crossentropy',
              metrics=odir_metrics,
              run_eagerly=True)

df_train = pd.read_feather(TRAINFT)

train_eyeprob = df_train.iloc[:,1].to_numpy()
train_eyeprob = np.array([list(i) for i in train_eyeprob])

train_ytrue = df_train.iloc[:,2].to_numpy()
train_ytrue = np.array([list(i) for i in train_ytrue])

X_train, X_test, y_train, y_test = train_test_split(
    train_eyeprob, train_ytrue, test_size=0.2, random_state=SEED)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

mc = ModelCheckpoint(EXP_NAME + '_best_model.h5', monitor='val_odir_metrics', mode='max', 
verbose=1, save_best_only=True)

model.fit(X_train, 
        y_train,
        steps_per_epoch= X_train.shape[0] // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[es, mc])
