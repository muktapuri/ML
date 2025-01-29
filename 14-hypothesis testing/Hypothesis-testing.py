# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:57:45 2025

@author: Mukta
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats
#provide statistical function
#stats contains a variety of statistical tests
from statsmodels.stats import descriptivestats as sd
#provide descriptve statistics tools , inclusind the sign_test.
from statsmodels.stats.weightstats import ztest
#used  for conducting z-tests on datatest

#1 sample sign test
#whenever there is a single and data is not normal
marks=pd.read_csv(r"C:\13_Linear-regression\Signtest.csv")

#normal QQ plot
import pylab
stats.probplot(marks.Scores, dist='norm',plot=pylab)
#create a QQ plot to visualy check if the data follows a normal distributippn
#test for normally
shapiro_test=stats.shapiro(marks.Scores)
#performs the shapiro-Wilk test for normality
#H0 (null hypothesis): the data is normally distributed
#H1 (alternate hypothesis): the data is not normaly distributed
#outputs a test statisctics and p -value
print("Shapiro Test:",shapiro_test)
#p_value is 0.024 <0.05 , data is not normal

#Descriptive statistics
print(marks.Scores.describe())
#mean=84.20 and median=89.00
#1-sample  sign test
sign_test_result=sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
print("sign Test Result:",sign_test_result)
#Result : p-value=0.82
#interpretation:
#H0 : The median of scores is equal to the mean of scores.
#H1 the median of scores is not equal to the mean scores.
#since the p-value  (0.82) is greater than 0.05, we fail to reject the null hypothesis
#concluusion: the median and mean of scores are statistically

#1 sample z-test
fabric = pd.read_csv(r"C:\13_Linear-regression\Fabric_data.csv")

#normality test
fabric_normality= stats.shapiro(fabric)
print("Fabric normality test:",fabric_normality)
#p value = 0.1460>0.05

fabric_mean=np.mean(fabric)
print("Mean Fabric Length:" ,fabric_mean)

# z-test
#z-test_result,p_val= ztest(fabric['Fabric_length'], value=150)
z_test_result, p_val = ztest(fabric['Fabric_length'], value=150)
print("Z-Test Result:" ,z_test_result,"P-value:",p_val)
#result : p-value =7.15 * 10^-6
#interpretation
#H0 : the mean fabric  length is 150
#H1 : the mean fabric length is not 150
#since the p-value is ectremely small(less than 0.05),we re
#conclusion:the mean fabric length significantly diffres from

#Mann-Whitney Test
fuel=pd.read_csv(r"C:\13_Linear-regression\mann_whitney_additive.csv")
fuel.columns=["Without_additive","With_additive"]

#normality test
print("Without  Additive Normality:" ,stats.shapiro(fuel.Without_additive))
#p=0.50>0.05: accept H0
print("With additive normality:",stats.shapiro(fuel.With_additive))
#0.04<0.05 : reject H0 data is not normal
#Mann-Whitney U Test
mannwhitney_result=stats.mannwhitneyu(fuel.Without_additive,fuel.With_additive)
print("Mann-Whitney test Result:", mannwhitney_result)
#result : p-value= 0.445
#interpretation:
#H0 : No diffrence in performance between without_additive and with_additive.
#H1 : A significant diffrence exists.
#since the  p-value (o.445) is greater than 0.05 , we fail to reject the null hypothesis
#conclusion adding fuel additive does not significanly impact performance.
#applies the mann-Whitney U Test to check if theres's a significant diffrence between
#H0 : No diffrence performance between two groups.
#H1 : Significant diffrence in performance.

#######################3
#Paired T-Test
sup=pd.read_csv(r"C:\13_Linear-regression\paired2.csv")

#normality test 
print("Supplier A Normality Test:",stats.shapiro(sup.SupplierA))
#p-value = 0.896199285 >0.05: fails to reject H0, data is normal
print("Supplier B Normality Test:",stats.shapiro(sup.SupplierB))
##p-value = 0.896199285 >0.05: fails to reject H0, data is normal
#paired T-test
t_test_result,p_val=stats.ttest.rel(sup['SupplierA'],sup['SupplierB'])
print("Paired T-test Result:",t_test_result,"P-value:",p_val)
#result: p-value = 0.00
#Interpretation:
#H0 : no significant diffrence in ttransaction time between supplier A and Supplier B
#H1 : A significant diffrence exists.
#since the p-value (0.00) is less than 0.05 , we reject the null hypothesis.
#conclusion:There is a significant diffrence in transation times between the two supplier

#two sample   T-test
offers = pd.read_excel(r"C:\13_Linear-regression\Promotion.xlsx")
offers.columns=['InterestRateWaiver','StandardPromotions']

#variance test
levene_test= scipy.stats.levene(offers.InterestRateWaiver,offers.StandardPromotions)
print('levene Test(variance):',levene_test)
#p-value =0.2875
#H0nn=variance equal
#H1= varince unwqual
#pvalue = o.2875>0.05 fail to reject null hypothesis (H0 is accepted)
#two -sample t-test
ttest_result=scipy.stats.ttest_ind(offers.InterestRateWaiver,offers.StandardPromotions)
print("two-sample t-test",ttest_result)
#result p-value=0.0242
#interpretation
#H0 : both offres have the same mean impact
#H1: the mean impact of the two offers are diffrent
# since the p-vale(0.0242) is less tha 0.05 , we reject the null hypothesis
#conclusion: theres is a significant diffrence two promotional offers

#moods median test
#objective: is the median of pooh ,and trigger are statistically equal or not
animal = pd.read_csv(r"C:\13_Linear-regression\animals.csv")

#normality test
print("pooh normality:", stats.shapiro(animal.Pooh))
#p-value = 0.0122
print("Pigiet Normality:",stats.shapiro(animal.Piglet))
#p-value=o.044
print("tigger normality",stats.shapiro(animal.Tigger))
#p-value= 0.0219
#H0 : data is normal
#H1 : data is not normal
#since all p value are less than 0.05 hence reject the null hypothesis
#data is not normal , hence mood's test
#median test
median_test_result=stats.median_test(animal.Pooh,animal.Piglet,animal.Tigger)
print("Mood's Median Test:", median_test_result)
#result: p-value = 0.185
#interpretation
#H0 : all grpoups have equal median
#H1

#one way ANOVA#VIMP
#objective :  is the transation of three suppliers are same
#significantly diffrent
contract= pd.read_excel(r"C:\13_Linear-regression\ContractRenewal_Data(unstacked).xlsx")
contract.columns=['Supp_A','Supp_B','Supp_B']

#normality test
print('Supp_A normality',stats.shapiro(contract.Supp_A))
print("supp_B normality :", stats.shapiro(contract.Supp_B))
print("supp_C normality :", stats.shapiro(contract.Supp_C))

#all p value are greater than 0.05
#we fail to reject the null hypothesis
#i.e is accepted means data is normal
#varinace test
levene_test= scipy.stats.levene(contract.Supp_A,contract.Supp_B,contract.Supp_C)
print("Levene test (varince):", levene_test)
#H0 : data is having equal vaarince
#H1: 
###############################################
#two proprtion z-test
import pandas as pd
import numpy as np
import scipy
from scipy import stats
#provide statistical function
#stats contains a variety of statistical tests
from statsmodels.stats import descriptivestats as sd
#provide descriptve statistics tools , inclusind the sign_test.
from statsmodels.stats.weightstats import ztest
#used  for conducting z-tests on datatest
#objective: there is a significant diffrence in soft drink consumption
#between adult and children
soft_drink= pd.read_excel(r"C:\13_Linear-regression\JohnyTalkers.xlsx")
#prac pandas dataframe
from  statsmodels.stats.proportion import proportion_ztest

#data preparaton
count =np.array([58,152])
nobs= np.array([480,740])
#the two proportion z-test compare the  proportion of two groups.here:
    #Count=[58,152]: the number of successes
    #(people consuming soft drinks) in each group (adults and children)
#nobs=[480,740]:the total number of observation
#the count and nob value come from summerizing the data
#about the soft drink consumption for adults and children
#here's how these are typicaly obtaine

#nobs : represents the total number of individuals surveyad
#in each group

#the total number of adults surveyed is 480.
#the total number of children surveyed is 740
#hence , nobs =[480,740]

#these values are often extracted from a daataset
#if your data is in a file(like 'JohnyTalkers.xlxs").
#you can calculate these valuse as follows:
    
import pandas as pd
file_path= r"C:\13_Linear-regression\JohnyTalkers.xlsx"
soft_drink_data=pd.read_excel(file_path)

#filter the data into adults and children categories
adults =soft_drink_data[soft_drink_data['Person']=='Adults']
children =soft_drink_data[soft_drink_data['Person']=='Children']


#count of successes (soft drink consumers) for eacg group
count_adults = adults[adults['Drinks']=='Purchased'].shape[0]
count_children = children[children['Drinks']=='Purchased'].shape[0]

#total observations for each group
nobs_adults= adults.shape[0]
nobs_children =children.shape[0]


#final arrays for  Z-test
count = [count_adults,count_children]
nobs = [nobs_adults,nobs_children]

print("Counts (Soft drink Consumers):",count)
print("TOtal Observations:",nobs)

#two side test
from statsmodels.stats.proportion import proportions_ztest

# Two-sided test
z_stat, p_val = proportions_ztest(count, nobs, alternative='two-sided')
print("Two-sided proportion Test:", z_stat, "P-Value:", p_val)

#z_stat,p_val = proportions_ztest(count,nobs,alternative='two-sided')

#print("Two-sided proportion Test:"),z_stat,"P-Value",p_val

#################################################33
#chi-square test
#objective: is  defective  proportion are independent of the country?
#the dataset contains two columns

#Defective: indictas whether an item is defective (likely binary,with 1 for defective and 0 for not defective)
#country:Specifies the country associated with the item("e.g. india)
#the dataset has 800 entries , and there are
#no missing values in either ,and there are  
#no missing values is either column, it appers to be designed to analyse

Bahman = pd.read_excel(r"C:\13_Linear-regression\Bahaman.xlsx")

#crosstabulation
count = pd.crosstab(Bahman["Defective"],Bahman["Country"])
count

#Chi-Square test
chi2_result=scipy.stats.chi2_contingency(count)
print("Chi-Square Test:",chi2_result)
#result : pvalue=0.6315
#interpretation
#H0: Defective proportion are independent  of the country
#H1: defective proportions depend on the country
#since the v-value(0.6315) is greatr htan 0.05,we accept the
