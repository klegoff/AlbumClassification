# -*- coding: utf-8 -*-
"""
Useful functions for the image part
"""

import copy
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns

from skimage.transform import downscale_local_mean, resize
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#############################################
#
#       Image part
#
#############################################

#### Sampling function
def naive_sampling(array, n):
    return copy.deepcopy(array[::n,::n,:])

def average_sampling(array, n):
    return copy.deepcopy(downscale_local_mean(array, factors=(n,n,1)).astype(int))

def clean_sampling(array,n):
    # sample using skimage resize function
    shape = array[::n,::n,:].shape[0]
    return resize(array, output_shape = (shape,shape), anti_aliasing=True)

try:
	from skimage.feature import hog #sometimes, this import gives an error, i didnt  find a solution, reset runtime if it happens
	#might be linked to the "GPU" excecution environmnet
	#### HOG feature computation
	def compute_hog(array, n, m = 3, p = 1):
		return copy.deepcopy(hog(array, orientations=n, pixels_per_cell=(m, m),cells_per_block=(p, p), visualize=True, multichannel=True)[1])
except:
	print("HOG import didnt work")


#### Data formatting 
def one_hot_encode(labels):
    """
    one hot encode the labels
    """
    # dict of correspondancies between labels and their indexes
    y_values = list(set(labels))

    y_dict = {}
    for i in range(len(y_values)):
        y_dict[y_values[i]] = i
    
    y = to_categorical([y_dict[elem] for elem in labels]) # one hot encoding
    classes = [np.argmax(elem) for elem in y]
    # cast to array
    return np.array(y).reshape(len(y),len(y_values),1), classes, y_dict
    
#### Confusion matrix plot
def plot_confusion(y_true,y_pred,y_dict,keep=[]):
    """
    plot confusion matrix
    functionnality to keep only certains classes (will remove both predicted labels and real labels)
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    labels= np.array(list(y_dict.keys()))
    # filter confusion matrix (keep only the chosen fields in keep variable)
    if len(keep) != 0:
        keep_lab_idx= [y_dict[elem] for elem in keep]
        red_cm = []
        for i in keep_lab_idx:
            red_cm.append(cm[i][keep_lab_idx])
        cm = np.array(red_cm)
        labels = labels[keep_lab_idx]
        
    #disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    sns.set(rc={'figure.figsize':(15,12)})
    ax = sns.heatmap(cm, xticklabels=labels,yticklabels=labels)
    ax.set(xlabel='Predicted genras', ylabel='Real genras')

def compute_class_weight(classes, n=1):
	"""
	input : 
	classes = list of classes index for our data
	n = power of the weights (n higher means we give much more importance to lower cardinality classes)
	returns : class_weight (dict)
	"""
	card_list = list(dict(Counter(classes)).values())
	max_ = max(card_list)
	card_list = [elem/max_ for elem in card_list] #compute cardinality ratio to the least popular class
	weight = [(1/elem)**n for elem in card_list]# take the inverse as weights to the power n
	class_weight = {}
	for i in range(len(weight)):
		class_weight[i] = weight[i]
	return class_weight
  
###### PREPROCESSING (apply filters to the raw images)

def preprocess_image_vectorized(row, sampling = "clean", n=16):
	"""
	to be applied to the msdi dataframe
	"""
	array = load_img(entry = row, msdi_path="")
	if sampling == "clean":
		return clean_sampling(array, n=16)
	if sampling == "average":
		return clean_sampling(array, n=16)
	if sampling == "hog":
		return compute_hog(array, n=16)
	if sampling == "naive":
		return naive_sampling(array, n=16)
  
#############################################
#
#       textual part
#
#############################################

def split_lyrics(path="data/msx_lyrics_genre.txt", save_path="data/msdi/lyrics/"):
	"""
	parse textual data and store the wordcount dataframe into json files
	savint in data/msdi/lyrics
	(initially : )
	"""
	#!mkdir $save_path

	with open(path, "r") as file:
		reader = file.readlines()
		for i in range(len(reader)):

			splitted = reader[i].replace("\n","").split(" ")
			track_id = splitted[0]
			style = splitted[1]

			word_count = pd.DataFrame([elem.split(":") for elem in splitted[2:]])

			vocab = {}

			word_count = pd.DataFrame([elem.split(":") for elem in splitted[2:]]) # get the word count feature
			word_count = pd.DataFrame(word_count.iloc[:,1].apply(int).values, index=word_count.iloc[:,0].apply(int).values) # transform into dataframe and cast into int
			word_count.iloc[:,0].to_json(save_path + track_id + ".json") # save into json



def load_lyrics(entry, msdi_path=""):
	"""
	get the lyrics data for the chosen entry
	not all the song have lyrics, so need to have a try/except
	"""
	try :
		pd.read_json(msdi_path + "lyrics/" + elem, typ="series")
		return x[entry['msd_track_id']]
	except:
		return None


if __name__ == '__main__':

	# preprocess lyrics data
	split_lyrics()