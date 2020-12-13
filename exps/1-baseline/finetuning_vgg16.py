from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 5
SEED = 13

vgg_conv = VGG16(weights='imagenet')

for layer in vgg_conv.layers: 
    layer.trainable = False

out = vgg_conv.get_layer('fc2').output
out = Dense(8, activation='softmax', name='predictions')(out)

model = Model(vgg_conv.input, out)

# We compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(validation_split=0.2)

train_gen = datagen.flow_from_directory("/work/ocular-dataset/ODIR-5K-Flow/train/", 
target_size=(244,244), batch_size=BATCH_SIZE, subset='training', seed=SEED)

val_gen = datagen.flow_from_directory("/work/ocular-dataset/ODIR-5K-Flow/train/", 
target_size=(244,244), batch_size=BATCH_SIZE, subset='validation', seed=SEED)

# fine-tune the model
history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.n // BATCH_SIZE)

loss_values = history.history['loss']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("./test.png")