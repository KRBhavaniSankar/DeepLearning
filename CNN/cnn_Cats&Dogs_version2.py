#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:20:07 2018

@author: bhavani
"""

#Importinng the keras libraries and packages
from keras.models import Sequential   #Initializing the NN as either Sequential or graph of layers.
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN 

classifier = Sequential()

#Step1- Convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3),activation='relu'))

#Step2 : Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding 2nd Convolution Layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Step3 : Flattening
classifier.add(Flatten())


#Step 4 - Fully Connected Layer

classifier.add(Dense(output_dim = 128,activation= "relu"))
classifier.add(Dense(output_dim = 1,activation= "sigmoid"))

#Compiling the CNN

classifier.compile(optimizer="adam",loss= "binary_crossentropy",metrics=["accuracy"])     #more than 2 , need to choose categorical_etropy

#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=10,
                        validation_data=test_set,
                        validation_steps=2000)
