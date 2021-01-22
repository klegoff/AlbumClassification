# -*- coding: utf-8 -*-
"""
Neural network models
"""

from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation
from keras import Sequential
from keras.metrics import CategoricalCrossentropy, Accuracy


import torch.nn as nn

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



#############################################
#
#       Textual part
#
#############################################

class lyrics_model(nn.Module):
  def __init__(self,embed_size, hidden_size):
    super().__init__()
    self.embed = nn.Embedding(len(lyrics), embed_size)
    self.seen_words_rnn = nn.GRU(embed_size,hidden_size,num_layers=1,bidirectional=False, batch_first=True)
    self.words_frequency_rnn = nn.GRU(embed_size,hidden_size,num_layers=1,bidirectional=False, batch_first=True)
    self.seen_words_dropout = nn.Dropout(0.3)
    self.words_frequency_dropout = nn.Dropout(0.3)
    self.linear = nn.Linear(hidden_size*2,out_features=len(label_vocab)) #*2 parce que j'ai deux couches 
  
  def forward(self,x_seen_words,x_words_frequency):
    sw_embed = self.embed(x_seen_words)
    wf_embed = self.embed(x_words_frequency)
    output_sw,hidden_sw = self.seen_words_rnn(sw_embed)
    output_wf,hidden_wf = self.words_frequency_rnn(wf_embed)
    sw_drop = self.seen_words_dropout(hidden_sw)
    wf_drop = self.words_frequency_dropout(hidden_wf)
    cat = torch.cat((sw_drop,wf_drop),-1) #concatene et renvoie un tenseur 
    return self.decision(cat.contiguous())


#### not functional : 
def perf(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = correct = num = 0
    for x, y in loader:
      with torch.no_grad():
        x_seen_words = x[:,:max_len] 
        x_words_frequency = x[:,max_len:]
        y_scores = model(x_seen_words,x_words_frequency)
        loss = criterion(y_scores, y)
        y_pred = torch.max(y_scores, 1)[1]
        correct += torch.sum(y_pred.data == y)
        total_loss += loss.item()
        num += len(y)
    return total_loss / num, correct.item() / num

def fit(model, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x_seen_words = x[:,:max_len] 
            x_words_frequency = x[:,max_len:]
            y_scores = model(x_seen_words,x_words_frequency)
            loss = criterion(y_scores, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
        print(epoch, total_loss / num, *perf(model, test_loader))

def predict(model,loader):
   output = []
   for x, y in loader:
     with torch.no_grad():
       x_seen_words = x[:, :max_len]
       x_words_frequency = x[:, :max_len]
       y_scores = self(x_seen_words, x_words_frequency)
       y_pred = y_scores > 0.5
       output.append(y_pred.int())
   return output

