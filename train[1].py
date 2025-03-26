from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
import numpy as np
import matplotlib.pyplot as plt

# image specifications
img_width, img_height = 224, 224  # dimension of the input image
train_data_dir = 'data/train'  # directory path for training data
valid_data_dir = 'data/valid'  # directory path for validation data
n_sample_train = 100  # number of train samples (train neural network)
n_sample_valid = 100  # number of validation samples (target neural network)
batch_size = 50

# input format check, not necessary but as a precaution
if K.image_data_format() == 'channels_last':
    input_shape = (img_width, img_height, 3)  # 3 means RGB
else:
    input_shape = (3, img_width, img_height)  # 3 means RGB (channels_first)

# design the network
model = models.Sequential()  # sequential model

model.add(Conv2D(32, (3, 3), input_shape=input_shape))  # convolutional layer 1 with 32 filters and kernel 3*3
model.add(Activation('relu'))  # relu activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling 1 filter size 2*2

model.add(Conv2D(32, (3, 3)))  # conv 2 filter 32, kernel 3*3
model.add(Activation('relu'))  # relu activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling 2 with filter 2*2

model.add(Conv2D(64, (3, 3)))  # conv 3 filter 64 kernel 3*3
model.add(Activation('relu'))  # relu activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling 3 with filter 2*2

model.add(Flatten())  # flatten layer

model.add(Dense(64))  # Dense layer 1
model.add(Activation('relu'))  # relu activation
model.add(Dropout(0.2))

model.add(Dense(1))  # output layer for binary classes
model.add(Activation('sigmoid'))  # sigmoid activation

# model compilation
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = History()

# augmentation of training data
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(valid_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

# training the model
epochs = 100
model.fit(train_generator,
          steps_per_epoch=n_sample_train // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=n_sample_valid // batch_size,
          callbacks=[history])

# saving the model
model.save('model.h5')

# visualizing training and validation accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# visualizing training and validation loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()