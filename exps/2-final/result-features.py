from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from os.path import isfile, join
from itertools import islice
import numpy as np
import pandas as pd
import csv

csvtrain = "/work/exps/train_gt.csv"
csvval = "/work/exps/val_gt.csv"

fttrain = "/work/exps/2-final/train_features.ft"
ftval = "/work/exps/2-final/val_features.ft"

image_base = "/work/ocular-dataset/ODIR-5K/Training Images/"

base_model = EfficientNetB3()

out = base_model.get_layer('top_dropout').output
out = Dense(8, activation='softmax', name='predictions')(out)

tmp_model = Model(base_model.input, out)

# We compile the model
tmp_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', 
metrics=['AUC'])

tmp_model.load_weights("ft_efficientnetb3_top_dropout_lr-4_best_model.h5")

tmp_out = tmp_model.get_layer('top_dropout').output
model = Model(tmp_model.input, tmp_out)

for csvfile, ftfile in ((csvtrain, fttrain), (csvval, ftval)):
    with open(csvfile, "r") as fp:
        csvreader = csv.reader(fp)
        lines = len(list(islice(csvreader,1,None)))

        feature_matrix = np.zeros((lines, 1536*2), np.float)
        y_true = np.zeros((lines, 8), np.int)
        patient_id = np.zeros([lines], np.int)
        count = 0

        fp.seek(0,0)

        csvreader = csv.reader(fp)

        for l in islice(csvreader,1,None):
            for side in ("left", "right"):
                image_path = join(image_base, l[0] + "_{}.jpg".format(side))

                if isfile(image_path) is True:
                    image = load_img(image_path, target_size=(300,300))
                    image = img_to_array(image)
                    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    image = preprocess_input(image)
                    yhat = model.predict(image)

                    if side == 'left':
                        feature_matrix[count][0:1536] = yhat
                    if side == 'right':
                        feature_matrix[count][1536:1536*2] = yhat

            patient_id[count] = int(l[0])
            y_true[count] = [int(label) for label in l[1:9]]
            count+= 1

        print("Converting DataFrame...")

        df = pd.DataFrame({'id':patient_id,
                'eyes_feature':[f for f in feature_matrix],
                'y_true':[y for y in y_true]})

        print("Saving DataFrame")

        df.to_feather(ftfile)
            
                    