from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

BATCH_SIZE = 16
QTDE_TEST = 348

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

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(flow, QTDE_TEST // BATCH_SIZE+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(flow.classes, y_pred)
print(cm)
print('Classification Report')
target_names = ['N', 'D', 'G', 'C', "A", "H", "M", "O"]
print(classification_report(flow.classes, y_pred, target_names=target_names))

#abc = plot_confusion_matrix(cm, target_names)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

abc = disp.plot().figure_
abc.savefig("confusion_matrix.png")