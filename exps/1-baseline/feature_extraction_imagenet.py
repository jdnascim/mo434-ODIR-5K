from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random


BASE_DATASET = "/work/ocular-dataset/"

BASE_IMAGES = os.path.join(BASE_DATASET, "ODIR-5K")

TRAIN_IMAGES = os.path.join(BASE_IMAGES, "Training Images")

FEATURES_BASE = os.path.join(BASE_DATASET, "features")
FEATURES_PATH = os.path.join(FEATURES_BASE, "vgg16-imagenet.ft")

#-----------------------------------------------

base_model = VGG16(weights='imagenet')
out = base_model.get_layer('fc2').output
model = Model(inputs=[base_model.input],outputs=out)

def features_extraction(frames_list, base_path, ftpath):

    frames_size = len(frames_list)

    feature_matrix = np.zeros([frames_size, 4096], dtype=np.float32)

    pct = 0

    print("Extracting Features")

    for i in range(frames_size):
        try:
            if frames_list[i] == "":
                continue

            src = os.path.join(base_path, frames_list[i])

            image = load_img(src, target_size=(224,224))

            # convert the image pixels to a numpy array
            image = img_to_array(image)

            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1],
                                    image.shape[2]))

            # prepare the image for the VGG model
            image = preprocess_input(image)

            yhat = model.predict(image)

            feature_matrix[i] = yhat

            if (i+1) % 1000 == 0:
                print(str(i+1) + " of " + str(frames_size))

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            return

    print("Converting to DataFrame...")

    df = pd.DataFrame({"path":frames_list,
        "feature_vector":[f for f in feature_matrix]})

    print("Saving DataFrame...")

    if os.path.isdir(os.path.dirname(ftpath)) is False:
        os.makedirs(os.path.dirname(ftpath))

    df.to_feather(ftpath)

imgs_list = os.listdir(TRAIN_IMAGES)

features_extraction(imgs_list, TRAIN_IMAGES, FEATURES_PATH)