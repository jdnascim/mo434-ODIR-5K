import pandas as pd
import numpy as np
import csv
import os

import ODIR_evaluation

CSVFILE = "/work/ocular-dataset/full_df.csv"
FEATHERFILE = "/work/ocular-dataset/features/vgg16-imagenet.ft"
XLSXFILE = "/work/ocular-dataset/ODIR-5K/data.xlsx"

TRAIN_GT = "/work/exps/train_gt.csv"
VAL_GT = "/work/exps/val_gt.csv"

VAL_GT_XLSX = "/work/exps/val_gt.xlsx"

GT_HEADER = ['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

SEED = 13

class ODIR_Dataset:
    def __init__(self):
        # read file with feature vectors
        df = pd.read_feather(FEATHERFILE)
        feature_dict = pd.Series(df.feature_vector.values, index=df.path).to_dict()

        # read file with labels from each eye
        csvfile = pd.read_csv(CSVFILE)
        labels_dict = pd.Series(csvfile.target.values, index=csvfile.filename).to_dict()

        # read files containing target for train and val patients
        df_train_gt = pd.read_csv(TRAIN_GT)
        df_val_gt = pd.read_csv(VAL_GT)

        train_set = set(df_train_gt['ID'].to_numpy())
        val_set = set(df_val_gt['ID'].to_numpy())

        self.X_train = np.zeros([len(train_set)*2], np.object)
        self.X_train_id = np.zeros([len(train_set)], np.object)
        self.y_train = np.zeros([len(train_set)*2, 8], np.int)
        self.X_val = np.zeros([len(val_set)*2], np.object)
        self.X_val_id = np.zeros([len(val_set)], np.object)

        xlsx = pd.read_excel(XLSXFILE)

        i_train, i_test = 0,0
        for _, row in xlsx.iterrows():
            if row['ID'] in train_set:
                self.X_train[2*i_train] = row['Left-Fundus']
                self.y_train[2*i_train] = [int(i) for i in labels_dict[row['Left-Fundus']][1:-1].split(', ')]
                self.X_train[2*i_train + 1] = row['Right-Fundus']
                self.y_train[2*i_train + 1] = [int(i) for i in labels_dict[row['Right-Fundus']][1:-1].split(', ')]

                self.X_train_id[i_train] = row['ID']

                i_train += 1
            elif row['ID'] in val_set:
                self.X_val[2*i_test] = row['Left-Fundus']
                self.X_val[2*i_test + 1] = row['Right-Fundus']

                self.X_val_id[i_test] = row['ID']

                i_test += 1
    
    def evaluate(self, y_val_res, resultfile):
        if resultfile in (TRAIN_GT, VAL_GT, VAL_GT_XLSX):
            raise Exception("resultfile with same names as gt files")

        if os.path.isfile(resultfile) is True:
            os.remove(resultfile)

        df_dict = dict()
        df_dict[GT_HEADER[0]] = self.X_val_id
        for i in range(1,len(GT_HEADER)):
            df_dict[GT_HEADER[i]] = y_val_res[:,i-1]

        df = pd.DataFrame(df_dict)

        df.to_csv(resultfile, index=False)

        ODIR_evaluation.ODIR_Evaluation(VAL_GT_XLSX, resultfile)