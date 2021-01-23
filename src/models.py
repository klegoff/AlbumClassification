# -*- coding: utf-8 -*-
"""
Neural network models
"""
from tensorflow.nn import softmax_cross_entropy_with_logits

from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation
from keras import Sequential
from keras.metrics import CategoricalCrossentropy, Accuracy

import torch.nn as nn

#############################################
#
#       Image part
#
#############################################

def image_model(input_shape):
    """
    return an instance of CNN for the images
    
    /!\ still need to find a way to implement a weighted loss (because imbalanced data)    
    """
    model = Sequential()
    # convolutional layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    # output layer : 15 labels to predict
    model.add(Dense(15, activation='softmax'))


    # compiling the sequential model
    #loss1 = weightedLoss(CategoricalCrossentropy, class_weight)
    #loss2 = custom_loss(CategoricalCrossentropy, class_weight)
    loss3 = CategoricalCrossentropy
    #loss4 = custom_loss
    loss5 = softmax_cross_entropy_with_logits
    model.compile(loss=loss5, metrics=Accuracy(), optimizer="Adam")
    #model.compile(loss=CategoricalCrossentropy, metrics = Accuracy(), optimizer="Adam")
    #model.compile(loss=, metrics=Accuracy(), optimizer='adam')

    return model



#############################################
#
#       Textual part
#
#############################################


class lyrics_model(nn.Module):
  def __init__(self,sequence_length,embed_size, hidden_size, genras):
    super().__init__()
    self.embed = nn.Embedding(sequence_length, embed_size)
    self.seen_words_rnn = nn.GRU(embed_size,hidden_size,num_layers=1,bidirectional=False, batch_first=True)
    self.words_frequency_rnn = nn.GRU(embed_size,hidden_size,num_layers=1,bidirectional=False, batch_first=True)
    self.seen_words_dropout = nn.Dropout(0.3)
    self.words_frequency_dropout = nn.Dropout(0.3)
    self.linear = nn.Linear(hidden_size*2,out_features=genras) #*2 parce que j'ai deux couches 
  
  def forward(self,x_seen_words,x_words_frequency):
    sw_embed = self.embed(x_seen_words)
    wf_embed = self.embed(x_words_frequency)
    output_sw,hidden_sw = self.seen_words_rnn(sw_embed)
    output_wf,hidden_wf = self.words_frequency_rnn(wf_embed)
    sw_drop = self.seen_words_dropout(hidden_sw)
    wf_drop = self.words_frequency_dropout(hidden_wf)
    cat = torch.cat((sw_drop,wf_drop),-1) #concatene et renvoie un tenseur 
    return self.decision(cat.contiguous())


#############################################
#
#       Audio part
#
#############################################

class mfcc_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.couche1 = nn.Sequential()
        self.couche1.add_module("Conv1", nn.Conv2d(in_channels = (num_rows,num_columns,num_channels), out_channels = 32 , kernel_size=(4,3)))
        self.couche1.add_module("ReLu1", nn.ReLU(inplace=True))
        self.couche1.add_module("MaxPooling1", nn.MaxPool2d(kernel_size(4,2)))

        self.couche2 = nn.Sequential()
        self.couche2.add_module("Conv2", nn.Conv2d(in_channels = 32, out_channels = 64 , kernel_size=(4,3)))
        self.couche2.add_module("ReLu2", nn.ReLU(inplace=True))
        self.couche2.add_module("MaxPooling2", nn.MaxPool2d(kernel_size(4,2)))

        self.couche3 = nn.Sequential()
        self.couche3.add_module("Conv3", nn.Conv2d(in_channels = 64, out_channels = 64 , kernel_size=(4,1)))
        self.couche3.add_module("ReLu3", nn.ReLU(inplace=True))
        self.couche3.add_module("MaxPooling3", nn.MaxPool2d(kernel_size(4,1)))

        self.couche4 = nn.Sequential()
        self.couche4.add_module("Conv4", nn.Conv2d(in_channels = 64, out_channels = 32 , kernel_size=(1,1)))
        self.couche4.add_module("ReLu4", nn.ReLU(inplace=True))
        self.couche4.add_module("MaxPooling4", nn.MaxPool2d(kernel_size(4,1)))
        self.couche4.add_module("Flatten", nn.Flatten())

        self.linear = nn.Sequential()
        self.linear.addd_module("Linear", nn.Linear(32*num_rows*num_columns*num_channels,num_labels))
        self.linear.add_module("Sigmoid", nn.Sigmoid(inplace=True))

    def forward(self,x):
        x_1 = self.couche1(x)
        x_2 = self.couche2(x_1)
        x_3 = self.couche3(x_2)
        x_4 = self.couche4(x_3)
        x = x_4.view(-1,32*num_rows*num_columns*num_channels)
        x = self.linear(x)
        return x


