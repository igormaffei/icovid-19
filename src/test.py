import os

covid_dir = os.path.join('../img_old/trainning/covid19')
notcovid_dir = os.path.join('../img_old/trainning/notcovid19')

#covid_dir = os.path.join('../img_resize/trainning/covid19')
#notcovid_dir = os.path.join('../img_resize/trainning/not_covid19')

print('total training covid-19 images:', len(os.listdir(covid_dir)))
print('total training not covid-19 images:', len(os.listdir(notcovid_dir)))

#import tensorflow as tf
#import keras_preprocessing
#from keras_preprocessing import image
import keras as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "../img_old/trainning/"
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "../img_old/test/"
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224, 224),
    class_mode='binary',
    shuffle=True,
    seed=42)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    class_mode='binary'
)

model = tf.models.Sequential([
     # This is the first convolution
    tf.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.layers.MaxPooling2D(2, 2),
    tf.layers.Dropout(0.15),
    # The second convolution
    tf.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.layers.MaxPooling2D(2, 2),
    tf.layers.Dropout(0.15),
    # The second convolution
    tf.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.layers.MaxPooling2D(2, 2),
    tf.layers.Dropout(0.20),
    # The second convolution
    tf.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.layers.MaxPooling2D(2, 2),
    tf.layers.Dropout(0.20),
    # The second convolution
    tf.layers.Conv2D(128, (3, 3), activation='relu', ),
    tf.layers.MaxPooling2D(2, 2),
    tf.layers.Dropout(0.25),
    # The third convolution
    # The fourth convolution
    tf.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.layers.MaxPooling2D(2, 2),
    tf.layers.Dropout(0.25),
    # Flatten the results to feed into a DNN
    tf.layers.Flatten(),
    #tf.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.layers.Dense(512, activation='relu'),
    tf.layers.Dense(2, activation='softmax')
])

# model = tf.models.Sequential([
#      # This is the first convolution
#     tf.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     tf.layers.MaxPooling2D(2, 2),
#     tf.layers.Dropout(0.25),
#     # The second convolution
#     tf.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.layers.MaxPooling2D(2, 2),
#     tf.layers.Dropout(0.25),
#     # The third convolution
#     tf.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.layers.MaxPooling2D(2, 2),
#     tf.layers.Dropout(0.25),
#     # The fourth convolution
#     tf.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.layers.MaxPooling2D(2, 2),
#     tf.layers.Dropout(0.25),
#     # Flatten the results to feed into a DNN
#     tf.layers.Flatten(),
#     #tf.layers.Dropout(0.5),
#     # 512 neuron hidden layer
#     tf.layers.Dense(512, activation='relu'),
#     tf.layers.Dense(2, activation='softmax')
# ])

model.summary()

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


history = model.fit(train_generator, epochs=25, validation_data=validation_generator, verbose=1)

model.save("../model/covid19.h5")

print("model saved...")

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

# import numpy as np
# from google.colab import files
# from keras.preprocessing import image
#
# uploaded = files.upload()
#
# for fn in uploaded.keys():
#     # predicting images
#     path = fn
#     img = image.load_img(path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     images = np.vstack([x])
#
#     classes = model.predict(images, batch_size=10)
#     print(fn)
#     print(classes)

#
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb
