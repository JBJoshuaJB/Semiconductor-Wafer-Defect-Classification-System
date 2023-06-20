import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import load_img ,img_to_array, to_categorical
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D 
from keras.callbacks import EarlyStopping
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from time import time
from PIL import Image

#1. INITIALIZATION
directory = 'On_SemiDataset'
classes = ['Contamination-Particle', 'Pattern defect', 'Probe Mark', 'Scratches', 'Others']
img_width, img_height = 64, 64
input_shape = (img_width, img_height, 3) #3 bytes colour 
epochs = 100

#2. IMAGE PREPARATION
train_gen = ImageDataGenerator(rescale = 1 / 255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               validation_split = 0.2)

train_data = train_gen.flow_from_directory(directory,
                                           target_size = (img_height, img_width),
                                           batch_size = 32,
                                           classes = classes,
                                           class_mode = 'categorical',
                                           shuffle = True,
                                           subset = 'training') #Set as training data

test_data = train_gen.flow_from_directory(directory,
                                          target_size = (img_height, img_width),
                                          batch_size = 32,
                                          classes = classes,
                                          class_mode = 'categorical',
                                          shuffle = True,
                                          subset = 'validation') #Set as validation data

#3. VISUALIZATION OF CLASSES SAMPLES 
contaminationparticle_img = tf.keras.utils.load_img('On_SemiDataset/Contamination-Particle/DP57572.1Y_09_9_A.png', color_mode = 'grayscale')
plt.imshow(contaminationparticle_img, cmap = 'gray')
plt.show()

patterndefect_img = tf.keras.utils.load_img('On_SemiDataset/Pattern defect/DP57572.1Y_06_76_B.png', color_mode = 'grayscale')
plt.imshow(patterndefect_img, cmap = 'gray')
plt.show()

probemark_img = tf.keras.utils.load_img('On_SemiDataset/Probe Mark/DP57572.1Y_09_96_C.png', color_mode = 'grayscale')
plt.imshow(probemark_img, cmap = 'gray')
plt.show()

scratch_img = tf.keras.utils.load_img('On_SemiDataset/Scratches/DP57572.1Y_09_95_D.png', color_mode = 'grayscale')
plt.imshow(scratch_img,cmap = 'gray')
plt.show()

others_img = tf.keras.utils.load_img('On_SemiDataset/Others/DP57572.1Y_09_82_E.png', color_mode = 'grayscale')
plt.imshow(others_img, cmap = 'gray')
plt.show()

#4. BUILD NN 
def model():
    
    #Create model
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding='same', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten()) #Flatten to feed into a dense layer
    model.add(Dense(100, activation = 'relu')) #100 neurons in the fully connected layer
    model.add(Dense(5, activation = 'softmax')) #5 output neurons for 5 classes with the softmax activation
   #model.add(Dropout(rate=0.2)) #Remove certain dense nodes to prevent overfitting

    return model

#5. BUILD MODEL 
model = model()
model.summary()
print(model.summary())
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

#6. FIT MODEL 
time0 = time()
history = model.fit(train_data, 
                    validation_data = test_data, 
                    epochs = epochs, 
                    batch_size = 200, 
                    verbose = 2)
print("\nTraining Time (in minutes) =",(time()-time0) / 60)

#7. SAVE MODEL 
model.save('defects_classification_model.h5')