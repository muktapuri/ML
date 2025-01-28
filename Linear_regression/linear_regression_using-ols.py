# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:44:30 2025

@author: Mukta
"""

import pandas as pd
import numpy as np
import seaborn as sns
wcat = pd.read_csv(r"C:\13_Linear-regression\wc-at.csv")
#EDA
wcat.info()
wcat.describe()
#average waist is 91.90 and min is 63.50 and max is 121
#averageAT is 101.89 and min is 11.44 and max is 253
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT,x=np.arange(1,110,1))
sns.distplot(wcat.AT)
#data is normal but right skewed
sns.boxplot(wcat.AT)
#no oitliers but right skewed
plt.bar(height=wcat.Waist,x=np.arange(1,110,1 ))
sns.distplot(wcat.Waist)
#Data is normal bimodal
plt.boxplot(wcat.Waist)
#No outliers but right skewed
####################################
#Bivariant analysis
plt.scatter(x=wcat.Waist,y=wcat.AT)
#Data is linearly scattred. direstion positive,strength:poor
#now let us check the correlation coefficient
np.corrcoef(wcat.Waist,wcat.AT)
#the correlation coeficient is 0.8185<0.85 hence the correlation is moderate
#let us check the direction of correlation
cov_output=np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
#635.91 is positive means correlation will be positive
##################################
#let us aaply to various  models and check the feasibility
import statsmodels.formula.api as smf
#forst simple linear regressio
model=smf.ols('AT~Waist',data=wcat).fit()
#Y is AT and X is waist
model.summary()
#R-squared = 0.67<0.85,  there  is scope of improvement
#p=00<0.85 hence acceptable
#bita-0=215.98
#bita-1=3.45
pred1=model.predict(pd.DataFrame(wcat.Waist))
pred1
######################
#regression line
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,'r')
plt.legend(['Predected Line','Observed data'])
plt.show()
###################
#error calculation
res1=wcat.AT-pred1
np.mean(res1)
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#32.76
#############################3
#let us try another function
#x=log(Waist)
plt.scatter(x=np.log(wcat.Waist),y=wcat.AT)
#Data is linearly scattered,direction positive, strength:poor
#now let us check the correlation coeficient
np.corrcoef(np.log(wcat.Waist),wcat.AT)
#the the correlation coeficient is 0.8185<0.85 hence the correlation 
#r=0.8217
model2=smf.ols('AT~np.log(Waist)',data=wcat).fit()
#Y is AT and X= log(Waist)
model2.summary()

pred2=model.predict(pd.DataFrame(wcat.Waist))
pred2
###############################
#regression line 
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,'r')
plt.legend(['Predicted Line','Observed_data_model2'])
plt.show()
########################
#error calculation
res2=wcat.AT-pred2
np.mean(res1)
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#32.49
###################################
#now let us make logY and X as is
plt.scatter(x=(wcat.Waist),y=np.log(wcat.AT))
#Data is linearly scattered,direction positive,strength:poor
np.corrcoef(wcat.Waist,np.log(wcat.AT))

model3=smf.ols('np.log(AT)~Waist',data=wcat).fit()

model3.summary()

pred3=model3.predict(pd.DataFrame(wcat.Waist))
pred3_at=np.exp(pred3)
pred3_at
########################
#Regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,'r')
plt.legend(['Predicted Line','Observed data_models'])
plt.show()
#######################
#error calculation
res3=wcat.AT-pred3_at

res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#38.52
#there are no significant changes ar r=0.8409
###########################33
#let us try another model
model4=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#y is log(AT) and x is Waist
model4.summary()
#R-Squared=0.779<0.85, there is scope of improvement
#p=0.000<0.5 hence acceptable
#bita-0 =-7.8241
#bita-1= 0.2289
pred4=model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at=np.exp(pred4)
pred4_at
############################
#regression line
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicted Line','Observed data_models'])
plt.show()
############
#error calculation
res4=wcat.AT-pred4_at

res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#32.24
#################################
#we have to genrelize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(wcat,test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))
plt.scatter(test.Waist.np.log(test.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#y is log(AT) and x Waist
final_model.summary()

test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at
############
train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at
########################3
#Evaluation on test data
test_res=test.AT-test_pred_at
test_sqr=test_res*test_res
test_mse=np.mean(test_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#############################
#Evalution on train data
train_res=train.AT-train_pred_at
train_sqr=train_res*train_res
train_mse=np.mean(train_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
############################
#test_rmse>train_rmse