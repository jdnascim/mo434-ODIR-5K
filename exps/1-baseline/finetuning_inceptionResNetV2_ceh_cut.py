from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

import pre_processing

BATCH_SIZE = 16
EPOCHS = 20
SEED = 13
LR = 1e-4

EXP_NAME = "ft_InceptionResNetV2_avg_pool_lr-4_ceh_cut"

PRE_PROC_FUNC = pre_processing.CEH_cut_pipeline

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

base_model = InceptionResNetV2(weights='imagenet')

for layer in base_model.layers: 
    layer.trainable = False

out = base_model.get_layer('avg_pool').output
out = Dense(8, activation='softmax', name='predictions')(out)

model = Model(base_model.input, out)

# We compile the model
model.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', 
metrics=['AUC'])

datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=PRE_PROC_FUNC)

train_gen = datagen.flow_from_directory("/work/ocular-dataset/ODIR-5K-Flow/train/", 
target_size=(244,244), batch_size=BATCH_SIZE, subset='training', seed=SEED)

val_gen = datagen.flow_from_directory("/work/ocular-dataset/ODIR-5K-Flow/train/", 
target_size=(244,244), batch_size=BATCH_SIZE, subset='validation', seed=SEED)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

mc = ModelCheckpoint(EXP_NAME + '_best_model.h5', monitor='val_auc', mode='max', 
verbose=1, save_best_only=True)

# fine-tune the model
history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.n // BATCH_SIZE,
        callbacks=[es, mc])

loss_values = history.history['loss']
loss_val_values = history.history['val_loss']
epochs = range(1, len(loss_values)+1)

fig, ax = plt.subplots()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.plot(epochs, loss_values, '-o', label='Training Loss')
plt.plot(epochs, loss_val_values, '-o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

textstr = "best val_auc: " + str(round(max(history.history["val_auc"]),4))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(EXP_NAME + ".png")