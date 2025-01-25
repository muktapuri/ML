# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 08:16:06 2025

@author: Mukta
"""

import pandas as pd
import numpy as np 
letters = pd.read_csv(r"C:\15-SVM\letterdata.csv")
'''
dataset typically used for handwritten character recognition
or related machine learning tasks. here's a breakdown of the structure

letter: represent the target class(the letter being identified,e.g.)
Feature: (xbox and yedgex) :these are numeric attribute
describing varois geomatric os statistical properties of the character
 xbox and ybox: x and y bounding box dimentions
 width and height: width and height of charcater's baounding box
 onpix: number of on pixels in the characters image
 xbar and y2bar and xybar: Variance and covariance of pixel intensities
 x2ybar

'''
#let us carry out EDA
a = letters.describe()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test =train_test_split(letters,test_size=0.2)
#let us split the data in terms x and y for both train and test data
train_X =train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X = test.iloc[:,1:]
test_y = test.iloc[:,0]
#kernel linear
model_linear = SVC(kernel="linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

#now let us check the accuracy =0.85675
np.mean(pred_test_linear==test_y)

#kernel rbf
model_rbf =SVC(kernel="rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
#now let us check the accuracy = 0.92275
np.mean(pred_test_rbf==test_y)

