from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

BATCH_SIZE = 16
EPOCHS = 10
SEED = 13

EXP_NAME = "ft_vgg16_fc2"

tf.random.set_seed(SEED)

vgg_conv = VGG16(weights='imagenet')

for layer in vgg_conv.layers: 
    layer.trainable = False

out = vgg_conv.get_layer('fc2').output
out = Dense(8, activation='softmax', name='predictions')(out)

model = Model(vgg_conv.input, out)

# We compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', 
metrics=['AUC'])

datagen = ImageDataGenerator(validation_split=0.2)

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

print(history.history.keys())

loss_values = history.history['loss']
auc_values = history.history['val_auc']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, label='Training Loss')
plt.plot(epochs, auc_values, label='Training AUC')
plt.xlabel('Epochs')
plt.legend()

plt.savefig(EXP_NAME + ".png")