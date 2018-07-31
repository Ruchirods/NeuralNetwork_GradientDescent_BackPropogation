# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:47:51 2018

@author: Tavanaei
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

def load_dataset(): # this returns 5 matrices/vectors
	# Training dataset
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
	# Training images (209*64*64*3)
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
	# Training labels (209*1)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

	# Test dataset
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
	# Test data (50*64*64*3)
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
	# Test labels (50*1)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

	#List of class names for test data "cat" "no-cat" not be used for computations
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(value):
    result = np.exp(value)/(1+np.exp(value))
    return result

def Sigmoidderivative(value):
    result=np.multiply((1-value),value)
    return result

def gradientdesecnt(train_set_x,train_set_y,test_set_x,test_set_y):
    w = np.random.randn((12288))
    mu = 0.05
    fileid=open("gradientdescent.txt",'w')
    fileid.write("Error for gradient descent"+'\n')
    for j in range(300):
        for i in range(209):
            Output = sigmoid(np.dot(train_set_x[:, i].T, w))
            fileid.write(str(train_set_y[:, i] - Output)+'\n')
            deltaW = (train_set_y[:, i] - Output) * train_set_x[:, i]
            w = w + mu * deltaW
    Output_test = sigmoid(np.dot(test_set_x.T, w))
    Output_test_Array = np.array(Output_test)
    Output_Result_Array = [(lambda i: 1 if i >= 0.5 else 0)(i) for i in Output_test_Array]
    Accuracy = np.sum(Output_Result_Array == test_set_y)*100 / 50.0
    print test_set_y
    print "Accurancy:", Accuracy

def backpropogation(train_set_x,train_set_y,test_set_x,test_set_y,path):
    w1 = np.random.randn(12288,952)
    w2=np.random.randn(952,1)
    mu = 0.000001
    fileID=open("backpropogation.txt",'w')
    fileID.write("Errors for BackPropogation:"+'\n')
    for i in range(209):
        inputvalue = np.matrix(train_set_x[:, i])
        Output1 = sigmoid(np.dot(w1.T, inputvalue.T))
        # Output1_mat = np.matrix(Output1)
        # print Output1.shape
        Output2 = sigmoid(np.dot(w2.T, Output1))
        # print Output2
        error = train_set_y[:, i] - Output2
        fileID.write(str(error)+'\n')
        deltaw = Output1 * (train_set_y[:, i] - Output2)
        # print "delta", deltaw.shape
        w2 = w2 + mu * deltaw
        # print "w2", w2.shape
        value = Sigmoidderivative(Output1)
        # print value.shape
        delta1 = np.multiply(np.multiply(w2, deltaw), value)
        # print "delta1", delta1.shape
        w1 = w1 + mu * np.dot(inputvalue.T, delta1.T)

    Output_test1 = sigmoid(np.dot(w1.T,test_set_x))
    Output_test2=sigmoid(np.dot(Output_test1.T,w2))
    Output_test_Array=np.array(Output_test2)
    Output_Result_Array = [(lambda i: 1 if i >= 0.5 else 0)(i) for i in Output_test_Array]
    Accuracy = np.sum(Output_Result_Array == test_set_y) * 100 / 50.0
    print "Accurancy:", Accuracy




train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

plt.figure()
plt.subplot(2,2,1)
plt.imshow(train_set_x_orig[25])
plt.subplot(2,2,2)
plt.imshow(train_set_x_orig[26])
plt.subplot(2,2,3)
plt.imshow(train_set_x_orig[27])
plt.subplot(2,2,4)
plt.imshow(train_set_x_orig[28])

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255.0
test_set_x = test_set_x_flatten / 255.0

print("Train Set: ")
print(train_set_x.shape)

print("Test Set: ")
print(test_set_x.shape)

print("Train Label: ")
print(train_set_y.shape)

print("Test Label: ")
print(test_set_y.shape)


# Start your code here
#Assignment part-1

gradientdesecnt(train_set_x,train_set_y,test_set_x,test_set_y)

#backpropogation(train_set_x,train_set_y,test_set_x,test_set_y)
#print Output_result
