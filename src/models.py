# -*- coding: utf-8 -*-
"""
Neural network models
"""

from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation
from keras import Sequential
from keras.metrics import CategoricalCrossentropy, Accuracy

#############################################
#
#       Image part
#
#############################################

def image_model():
    """
    create an instance of CNN for the images
    """
    model = Sequential()
    # convolutional layer
    model.add(Conv2D(30, (5,5), input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    #model.add(MaxPool2D(pool_size=(3,3)))

    model.add(Conv2D(20, (4,4), input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    #model.add(MaxPool2D(pool_size=(3,3)))

    model.add(Conv2D(15, (3,3), input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(Dropout(0.6))
    model.add(MaxPool2D(pool_size=(5,5)))
    model.add(Flatten())

    # output layer : 15 labels to predict
    model.add(Dense(15, activation='softmax'))


    # compiling the sequential model
    #loss1 = weightedLoss(CategoricalCrossentropy, class_weight)
    #loss2 = custom_loss(CategoricalCrossentropy, class_weight)
    loss3 = CategoricalCrossentropy
    loss4 = custom_loss
    loss5 = softmax_cross_entropy_with_logits
    model.compile(loss=loss5, metrics=Accuracy(), optimizer="Adam")
    #model.compile(loss=CategoricalCrossentropy, metrics = Accuracy(), optimizer="Adam")
    #model.compile(loss=, metrics=Accuracy(), optimizer='adam')

    return model
