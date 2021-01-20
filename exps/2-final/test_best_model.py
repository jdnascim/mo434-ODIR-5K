from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = EfficientNetB3(weights='imagenet')

out = base_model.get_layer('top_dropout').output
out = Dense(8, activation='softmax', name='predictions')(out)

model = Model(base_model.input, out)

# We compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', 
metrics=['AUC'])

model.load_weights("ft_efficientnetb3_top_dropout_lr-4_best_model.h5")

datagen = ImageDataGenerator()
flow = datagen.flow_from_directory("/work/ocular-dataset/ODIR-5K-Flow/fake-test")

loss, auc = model.evaluate(flow)

print("loss: ", str(loss))
print("auc: ", str(auc))

