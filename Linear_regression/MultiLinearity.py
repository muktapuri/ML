# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:11:34 2025

@author: Mukta
"""

import numpy as np
import pandas as pd
cars=pd.read_csv(r"C:\13_Linear-regression\Cars.csv")
#Exploratory data analysis
# 1. Measure the central tendency
# 2. measure the dispersion
# 3. third moment business decision
# 4. fourth moment business decision
# 5. probablity distribution
# 6. graphical representation (histogram, boxplot)

cars.describe()

#graphical representation 
import matplotlib.pyplot as plt
plt.bar(height=cars.HP, x=np.arange(1,82,1))
plt.hist(cars.HP)
plt.boxplot(cars.HP)


# There are sevtal outlire in HP column
# similar operations are expected for other three column


#Now let us plot joint plot , joint plot is to show scatter plot
#histogram

import seaborn as sns
sns.jointplot(x=cars['HP'], y=cars['MPG'])

#Now let us plot count plot
plt.figure(1, figsize=(16,10))
sns.countplot(cars['HP'])

# count plot shows how many times the each value occured
# 92 value occured 7 times

# QQ plot

from scipy import stats
import pylab

stats.probplot(cars.MPG, dist='norm', plot=pylab)
plt.show()

# MPG data is normaly distributed
# there are 10 scatter plot need to be loted , one by one 

# to plot , so we can use pair plot

import seaborn as sns
sns.pairplot(cars.iloc[:,:])

# you can check the collinarity problem between the input and output
# you can check plot between SP and HP. they are strongly correlated

#same way you can check WT and VOL, it is also strongly correlated


# now let us check r value betwen variables
cars.corr()
#you can check SP and HP,r value is 0.97 and same way
#you can check WT and VOL ,it has got 0.999 which is greater

#now although we observed strongly correlated pairs
#still we will go far linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
# R square value observed is 0.771<0.85
#p-values of WT and VOL is 0.814 and 0.556 which is vary high
#it means it is greater than 0.05,WT and VOL columns
#we need to ignore
#or delete .Instead deleting 81 entries,let us check
#row wise outliers
#identifying is there any influemtial value.
#to check ypu can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
#76 is the outliers
#go to data frame and check 76 the entry
#let us delete that entry
cars_new=cars.drop(cars.index[[76]])
#again apply regression
#R square value is 0.819 but p values are same ,hence not solving
#now next option is delete the column but
#quetion is which option is to be deleted
#we have already checked correlation factor r
#VOL has got -0.529 and WT =-0.526
#WT is less hence can be delted

#another approch is to check the  collinearity,rsquare is give
#that value is will have to apply regression w.r.t.x1 and input
#as x2,x3 and x4 so on so forth
rsq_hp=smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp=1/(1-rsq_hp)

#VIF is variance influential factor ,calculating VIF helps to
#of x1 w.r.t x2,x3 nad x4

rsq_wt=smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt=1/(1-rsq_wt)

rsq_vol=smf.ols('VOL~HP+WT+SP',data=cars).fit().rsquared
vif_vol=1/(1-rsq_vol)

rsq_sp=smf.ols('SP~HP+WT+VOL',data=cars).fit().rsquared
vif_sp=1/(1-rsq_sp)
#cif_wt=639.53 and vif_vol=638.80 hemnce vif_wt is greater

#storing the values in dataframe
d1={'Vatiable':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame


#let us drop WT and apply correlation to remaining three
final_ml=smf.ols('MPG~VOL+SP+HP',data=cars).fit()
final_ml.summary()
#e-square is 0.770 and p values 0.00,0.012<0.85

#prediction
pred=final_ml.predict(cars)

#QQ plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

#htis QQplot is on residual whic is obtainedn on training data
#eroors are obtaine on test data
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

sns.residplot(x=pred,y=cars.MPG,lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()
#residual plot are used to check whether the 
#error are independen or not

#let us plot the influence plot
sm.graphics.influence_plot(final_ml)
##we haave taken cars instead car_new data,hence 76 is reflecated
#again in influemtial data

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train,cars_test=train_test_split(cars, test_size=0.2)
#Preparing the model on train data
model_train=smf.ols('MPG~VOL+SP+HP',data=cars_train).fit()
model_train.summary()
test_pred=model_train.predict(cars_test)
#test errors
test_error=test_pred-cars_test.MPG
test_rmse=np.sqrt(np.mean(test_error*test_error))
test_rmse




