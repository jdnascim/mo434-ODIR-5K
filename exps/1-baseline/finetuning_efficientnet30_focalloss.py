from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import cohen_kappa_score, f1_score
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import datetime

BATCH_SIZE = 16
EPOCHS = 20
SEED = 13
LR = 1e-4

def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
	
	
EXP_NAME = "ft_efficientnetb3_top_dropout_lr-4"

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

base_model = EfficientNetB3(weights='imagenet')

for layer in base_model.layers: 
    layer.trainable = False

out = base_model.get_layer('top_dropout').output
out = Dense(8, activation='softmax', name='predictions')(out)

model = Model(base_model.input, out)

# We compile the model
model.compile(optimizer='nadam', loss=focal_loss(gamma = 2.0, alpha=0.2), metrics=['AUC'])
datagen = ImageDataGenerator(validation_split=0.2)

train_gen = datagen.flow_from_directory("train/", 
target_size=(244,244), batch_size=BATCH_SIZE, subset='training', seed=SEED)

val_gen = datagen.flow_from_directory("train/", 
target_size=(244,244), batch_size=BATCH_SIZE, subset='validation', seed=SEED)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

mc = ModelCheckpoint(EXP_NAME + '_best_model.h5', monitor='val_auc', mode='max', 
verbose=1, save_best_only=True)

%load_ext tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.n // BATCH_SIZE,
        callbacks=[ mc, tensorboard_callback])
		
%tensorboard --logdir logs/fit
loss_values = history.history['auc']
loss_val_values = history.history['val_loss']
epochs = range(1, len(loss_values)+1)

loss_values = history.history['loss']
loss_val_values = history.history['val_loss']
epochs = range(1, len(loss_values)+1)

fig, ax = plt.subplots()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

#plt.ylim(0,1)
plt.plot(epochs, loss_values, '-o', label='Training Loss')
plt.plot(epochs, loss_val_values, '-o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss per Epochs")
plt.legend()

textstr = "best val_auc: " + str(round(max(history.history["val_auc"]),4))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(EXP_NAME + ".png")