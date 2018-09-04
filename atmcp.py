# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:02:07 2018

@author: hugof
"""
                   
import json
import pandas as pd

import numpy as np

with open(r"C:\Users\hugof\OneDrive\Documents\Kaggle\train.json") as f:
    data = json.load(f)
    
train = pd.DataFrame(data)

with open(r"C:\Users\hugof\OneDrive\Documents\Kaggle\test.json") as f:
    data = json.load(f)
    
test = pd.DataFrame(data)

#%%

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
    
#%%
    
lexicon_ingredients,lexicon_cuisine=create_lexicon(train)
binary_ingredients,binary_cuisine=create_arrays(lexicon_ingredients,lexicon_cuisine,train)

train_x,train_y=binary_ingredients[:int(0.8*len(binary_ingredients))],binary_cuisine[:int(0.8*len(binary_cuisine))]

test_x,test_y=binary_ingredients[int(0.8*len(binary_ingredients)):len(binary_ingredients)],binary_cuisine[int(0.8*len(binary_cuisine)):len(binary_cuisine)]

#%%

import matplotlib.pyplot as plt

rate=0.2

train_x_bis,train_y_bis,test_x_bis,test_y_bis=train_x[:int(rate*len(train_x))],train_y[:int(rate*len(train_y))],test_x[:int(rate*len(test_x))],test_y[:int(rate*len(test_y))]

import tensorflow as tf

n_nodes_hl1=1000
n_nodes_hl2=1000
n_nodes_hl3=500
n_nodes_hl4=500

dropout_rate=0.9

n_classes=len(train_y[0])
batch_size=100

x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')

def neural_network_model(data):
    
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)
    l1=tf.nn.dropout(l1, dropout_rate)
    
    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)
    l2=tf.nn.dropout(l2, dropout_rate)
    
    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3=tf.nn.relu(l3)
    l3=tf.nn.dropout(l3, dropout_rate)
    
    l4=tf.add(tf.matmul(l3,hidden_4_layer['weights']),hidden_4_layer['biases'])
    l4=tf.nn.relu(l4)
    l4=tf.nn.dropout(l4, dropout_rate)
    
    output=tf.add(tf.matmul(l4,output_layer['weights']),output_layer['biases'])

    return(output)
    
def train_neural_network(x):
    Acc=[]
    prediction=neural_network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs=40
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss=0
            
            i=0
            while i<len(train_x_bis):
                start=i
                end=i+batch_size
                
                batch_x=np.array(train_x_bis[start:end])
                batch_y=np.array(train_y_bis[start:end])
                
                _ , c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epoch_loss+=c
                i+=batch_size
                
            print('Epoch',epoch,'completed out of', hm_epochs,'loss',epoch_loss)
                
            
            correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            
            accuracy=tf.reduce_mean(tf.cast(correct,'float'))
            a=accuracy.eval({x:test_x_bis,y:test_y_bis})
            Acc.append(a)
            print('Accuracy :',a)
            
    plt.plot(Acc)
            
train_neural_network(x)



        
        
        
        
    
    