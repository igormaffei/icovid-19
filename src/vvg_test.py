import os

#covid_dir = os.path.join('../img_old/trainning/covid19')
#notcovid_dir = os.path.join('../img_old/trainning/notcovid19')

covid_dir = os.path.join('../img_new2/trainning/covid19')
notcovid_dir = os.path.join('../img_new2/trainning/not_covid19')

print('total training covid-19 images:', len(os.listdir(covid_dir)))
print('total training not covid-19 images:', len(os.listdir(notcovid_dir)))


from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

train_path = '../img_new2/trainning/'
test_path = '../img_new2/test/'
batch_size = 16
image_size = 224


train_datagen = ImageDataGenerator(validation_split=0.3,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
                        directory=train_path,
                        target_size=(image_size,image_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        color_mode='rgb',
                        shuffle=True)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
                        directory=test_path,
                        target_size=(image_size, image_size),
                        color_mode='rgb',
                        shuffle=False,
                        class_mode='categorical',
                        batch_size=1)


import keras
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16

# Load the VGG model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

print(base_model.summary())

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# # Create the model
model = keras.models.Sequential()

# # Add the vgg convolutional base model
model.add(base_model)

# # Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))

# # Show a summary of the model. Check the number of trainable parameters
print(model.summary())

# # Compile the model
from keras.optimizers import SGD

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-3),
              metrics=['accuracy'])

# Start the training process
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n / batch_size,
    validation_data=test_generator,
    epochs=50)

model.save('../model/covid_vgg16.h5')

print("model saved...")

# summarize history for accuracy
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

print("print result")
plt.show()
print(".: FINISH :.")
