# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:02:07 2018

@author: hugof
"""
                   
import json
import pandas as pd

import numpy as np
import os 




with open(r"C:\Users\Tim\Desktop\tensorflow\bouffekaggle\all\train.json") as f:
    data = json.load(f)
    
train = pd.DataFrame(data)

with open(r"C:\Users\Tim\Desktop\tensorflow\bouffekaggle\all\test.json") as f:
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
import tqdm as tqdm

rate=1

#les listes bis sont la pour pas avoir a relancer le bloc au dessus a chaque changement de rate

train_x_bis,train_y_bis,test_x_bis,test_y_bis=train_x[:int(rate*len(train_x))],train_y[:int(rate*len(train_y))],test_x[:int(rate*len(test_x))],test_y[:int(rate*len(test_y))]


#%%
import tensorflow as tf

#changement du nombre de neurones par couche

n_nodes_hl1=3000
n_nodes_hl2=3000
n_nodes_hl3=2500
n_nodes_hl4=2000
n_nodes_hl5=1000

#probabilite de garder le neurone lors du dropout (evite aux poids d aller vers l infini)
dropout_rate=0.99

n_classes=len(train_y[0])

#taille des batchs
batch_size=128

x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')

def neural_network_model(data):
    
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl5])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}
#    hidden_5_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_nodes_hl5])),
#                   'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl5,n_classes])),
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
    
#    l5=tf.add(tf.matmul(l4,hidden_5_layer['weights']),hidden_5_layer['biases'])
#    l5=tf.nn.relu(l5)
#    l5=tf.nn.dropout(l5, dropout_rate)
    
    output=tf.add(tf.matmul(l4,output_layer['weights']),output_layer['biases'])

    return(output)
    
def train_neural_network(x):
    #Acc c est l historique des accuracy
    Acc=[]
    Loss=[]
    #def du RN de la fct de cout et de la descente du gradient
    prediction=neural_network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer=tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)
    
    hm_epochs=40
    batch_count = len(train_x_bis)//batch_size
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss=0
            
            
            j=0
            #on train par batch, ca peut etre bien de randomiser le choix des indices de batch sachant que 
            #pour le moment on les prend juste les uns a la suite des autres 
            print ('-'*15, 'Epoch %d' % epoch, '-'*15)
            for j in tqdm.tqdm(range(batch_count)):
                start=j*batch_size
                end=start+batch_size
                
                batch_x=np.array(train_x_bis[start:end])
                batch_y=np.array(train_y_bis[start:end])
                
                _ , c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epoch_loss+=c
                
            #on peut recuperer l historique des loss ici de la meme maniere qu avec accuracy   
            print('Epoch',epoch,'completed out of', hm_epochs,'loss',epoch_loss)
            Loss.append(epoch_loss)  
            
            correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            
            accuracy=tf.reduce_mean(tf.cast(correct,'float'))
            a=accuracy.eval({x:test_x_bis,y:test_y_bis})
            Acc.append(a)
            print('Accuracy :',a)
    #le trace de la courbe, ce qui pourrait etre bien c est d avoir un trace qui s actualise (dans l absolue
    #je sais pas faire)
    plt.plot(Acc)
    plt.plot(Loss/max(Loss))
    return( Acc,Loss )
            
Acc, Loss = train_neural_network(x)

#%%
import numpy

train_x_ter = numpy.array([numpy.asarray(i) for i in train_x_bis])
train_y_ter = numpy.array([numpy.asarray(i) for i in train_y_bis]) 
test_x_ter = numpy.array([numpy.asarray(i) for i in test_x_bis])
test_y_ter = numpy.array([numpy.asarray(i) for i in test_y_bis])



#%%
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import metrics
from keras.optimizers import Adam
from keras import initializers
import numpy

numpy.random.seed(10)

def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def createModel(optimizer):
    model = Sequential()
    
    model.add(Dense(700,input_dim=len(train_x[0]), activation='relu'))
    
    model.add(Dropout(0.3))
    
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(20, activation='relu'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy])
    return model

def train(epochs=1, batch_size=128):
    
    
    x_train, y_train, x_test, y_test = train_x_ter, train_y_ter, test_x_ter, test_y_ter
    
    batch_count = len(x_train) // batch_size
    
    adam = get_optimizer()
    model = createModel(adam)
    
#    for e in range(1, epochs+1):
#        print ('-'*15, 'Epoch %d' % e, '-'*15)
#            
#        for j in tqdm.tqdm(range(batch_count)):
#            start=j*batch_size
#            end=start+batch_size
#            
#            batch_x=np.array(x_train[start:end])
#            batch_y=np.array(y_train[start:end])
#            
#            model.train_on_batch(batch_x,batch_y)
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test), verbose = 2)
    plt.figure(0)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
        ##print("\n%s: %.2f%%" % (model.metrics_names[1], 100*scores[1]))
        
train(15,100)
                
            
    
    
    
