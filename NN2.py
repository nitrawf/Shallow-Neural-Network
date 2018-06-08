# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:52:57 2018

@author: anish
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def loaddata():
    with open("set.csv",newline='') as csvfile:
        lines=csv.reader(csvfile,delimiter=",",quoting=csv.QUOTE_NONE, quotechar='')
        for x in lines:
            templist=[]
            for i in range(len(x)-1):
                if x[i]=="Female":
                    x[i]=0
                elif x[i]=="Male":
                    x[i]=1
                templist.append(float(x[i]))
            X_list.append(templist)
            if x[len(x)-1]=="1":
                Y_list.append([0])
            else:
                Y_list.append([1])

def layer_sizes(X, Y):  
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_y)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.rand(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.rand(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}   
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
   
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache   

def compute_cost(A2, Y, parameters):    
    m = Y.shape[1] # number of example
    logprobs = np.dot(Y,np.log(A2).T)+np.dot(1-Y,np.log(1-A2).T)
    cost = (-1/m)*sum(logprobs)
    cost= np.squeeze(cost,axis=0) 
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}  
    return grads


def update_parameters(parameters, grads, learning_rate = 1.2):   
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]   
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]   
    W1 = W1-dW1
    b1 = b1-db1
    W2 = W2-dW2
    b2 = b2-db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]    
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs=[]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)        
        grads = backward_propagation(parameters, cache, X, Y)        
        parameters = update_parameters(parameters, grads)
        if i%100==0:
            costs.append(cost)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return costs,parameters

def predict(parameters, X):  
    A2, cache = forward_propagation(X, parameters)
    predictions=(A2>0.5)       
    return predictions



X_list=[]
Y_list=[]                
loaddata()
X=np.array(X_list).T
Y=np.array(Y_list).T
Xmeans=np.mean(X,axis=1).reshape(X.shape[0],1)
Xdevs=np.std(X,axis=1).reshape(X.shape[0],1)
X=(X-Xmeans)/Xdevs
X_trainset=X[:,0:400]
Y_trainset=Y[:,0:400]
X_testset=X[:,400:]
Y_testset=Y[:,400:]
n_h=7
costs,parameters = nn_model(X_trainset, Y_trainset, n_h , num_iterations = 10000, print_cost=True)
costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Hidden Layers: %d"%n_h)
plt.show()


predictions = predict(parameters, X_trainset)
print ('Accuracy for training set: %d' % float((np.dot(Y_trainset,predictions.T) + np.dot(1-Y_trainset,1-predictions.T))/float(Y_trainset.size)*100) + '%')

predictions = predict(parameters, X_testset)
print ('Accuracy for test set: %d' % float((np.dot(Y_testset,predictions.T) + np.dot(1-Y_testset,1-predictions.T))/float(Y_testset.size)*100) + '%')