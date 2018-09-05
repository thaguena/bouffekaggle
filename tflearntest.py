# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:30:08 2018

@author: hugof
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../"))

import json

with open("../Tim/Desktop/tensorflow/bouffekaggle/all/train.json") as f:
    data = json.load(f)
    
train = pd.DataFrame(data)

with open("../Tim/Desktop/tensorflow/bouffekaggle/all/test.json") as f:
    data = json.load(f)
    
test = pd.DataFrame(data)

def create_lexicon(data):
    
    lexicon_ingredients=[]
    lexicon_cuisine=[]
    
    for ingredients in data['ingredients'] :
        lexicon_ingredients+=ingredients
    
    for cuisine in data['cuisine']:
        lexicon_cuisine+=[cuisine]
        
    return(list(set(lexicon_ingredients)),list(set(lexicon_cuisine)))
    

def create_arrays(lexicon_ingredients,lexicon_cuisine,data):
    
    binary_ingredients=[]
    
    for recipe in data['ingredients']:
        binary_recipe=[0]*len(lexicon_ingredients)
        for ingredient in recipe:
            if ingredient in lexicon_ingredients :
                binary_recipe[lexicon_ingredients.index(ingredient)]=1
        binary_ingredients.append(binary_recipe)
        
    binary_cuisine=[]
        
    for cuisine in data['cuisine']:
        binary_recipe=[0]*len(lexicon_cuisine)
        if cuisine in lexicon_cuisine:
            binary_recipe[lexicon_cuisine.index(cuisine)]=1
        binary_cuisine.append(binary_recipe)
        
    return(binary_ingredients,binary_cuisine)

lexicon_ingredients,lexicon_cuisine=create_lexicon(train)
binary_ingredients,binary_cuisine=create_arrays(lexicon_ingredients,lexicon_cuisine,train)

train_x,train_y=binary_ingredients,binary_cuisine


# Any results you write to the current directory are saved as output.

#%%
import tflearn
from tensorflow import reset_default_graph
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout,fully_connected
from tflearn.layers.estimator import regression

reset_default_graph()

convnet=input_data(shape=[None,len(train_x[0])],name='input')

convnet=fully_connected(convnet,2000,activation='relu')
convnet=dropout(convnet,0.7)

convnet=fully_connected(convnet,2000,activation='relu')
convnet=dropout(convnet,0.7)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.7)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.7)
#
#convnet=fully_connected(convnet,1024,activation='relu')
#convnet=dropout(convnet,0.8)

convnet=fully_connected(convnet,20,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.002,loss='categorical_crossentropy',name='targets')

model=tflearn.DNN(convnet,max_checkpoints=1,tensorboard_dir="../Tim/Desktop/tensorflow/bouffekaggle/tflearn_logs/", tensorboard_verbose=0)

model.fit({'input':train_x},{'targets':train_y},n_epoch=1,
          snapshot_step=1000,show_metric=True,run_id='cuisine')

model.save('../Tim/Desktop/tensorflow/bouffekaggle/tflearncnn.model')

model.load('../Tim/Desktop/tensorflow/bouffekaggle/tflearncnn.model')

#%%

def create_arrays_test(lexicon_ingredients,data):
    
    binary_ingredients=[]
    
    for recipe in data['ingredients']:
        binary_recipe=[0]*len(lexicon_ingredients)
        for ingredient in recipe:
            if ingredient in lexicon_ingredients :
                binary_recipe[lexicon_ingredients.index(ingredient)]=1
        binary_ingredients.append(binary_recipe)
        
    return(binary_ingredients)
    
binary_ingredients_test=create_arrays_test(lexicon_ingredients,test)
test['bool_ing']=binary_ingredients_test


def results(test,model,lexicon_cuisine):
    results=[]
    for recipe in test['bool_ing']:
        index=np.argmax(model.predict([recipe])[0])
        results.append(lexicon_cuisine[index])
    return(results)

liste_results=results(test,model,lexicon_cuisine)

test['cuisine']=liste_results

my_submission = pd.DataFrame({'id': test.id, 'cuisine': test.cuisine})
# you could use any filename. We choose submission here
my_submission.to_csv('../Tim/Desktop/tensorflow/bouffekaggle/sub/submission.csv', index=False)