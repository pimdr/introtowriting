# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:47:00 2022

@author: pimde
"""

#order:  [0bmi,  1gender,  2step_est,  3step_tot,  4stap_avg,  5ipaqtot1,  6error,  7relative error, 8ipaqtot1/step_tot]

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn import datasets, linear_model
import statsmodels.api as sm
from scipy import stats

df = pd.read_excel('Exceldata.xlsx') # can also index sheet by name or fetch all sheets

bmi = np.array(df['bmi']).reshape(290,1)
gender = np.array(df['gender']).reshape(290,1)
stap_est = np.array(df['stap_est']).reshape(290,1)

stap_1 = np.array(df['stap_app_1_aantal']).reshape(290,1)
stap_2 = np.array(df['stap_app_2_aantal']).reshape(290,1)
stap_3 = np.array(df['stap_app_3_aantal']).reshape(290,1)
stap_4 = np.array(df['stap_app_4_aantal']).reshape(290,1)
stap_5 = np.array(df['stap_app_5_aantal']).reshape(290,1)
stap_6 = np.array(df['stap_app_6_aantal']).reshape(290,1)
stap_7 = np.array(df['stap_app_7_aantal']).reshape(290,1)
ipaqtot1 = np.array(df['ipaqtot1']).reshape(290,1)

stap_tot = np.add(stap_1,stap_2)
stap_tot = np.add(stap_tot,stap_3)
stap_tot = np.add(stap_tot,stap_4)
stap_tot = np.add(stap_tot,stap_5)
stap_tot = np.add(stap_tot,stap_6)
stap_tot = np.add(stap_tot,stap_7)

stap_avg = stap_tot/7
stap_avg = stap_avg.astype(int)


combine = np.concatenate((bmi,gender,stap_est,stap_tot,stap_avg,ipaqtot1),axis=1)

#print(combine.shape)
#print(combine[0][2])
#print(type(combine[0][2]))

#-----------------------------------remove nan values ------------------------------
indexlist_remove = []
for j in range(len(stap_1)):
    for i in range(6):
        
        if isinstance(combine[j][i], float) == True:   
            var = False
            var = math.isnan(combine[j][i])
            #print(var)
            if var == True:
                indexlist_remove.append(j)
            
        
indexlist_remove = list(dict.fromkeys(indexlist_remove))
indexlist_remove.reverse()
#print(indexlist_remove)    

#print(len(indexlist_remove))
for i in range(len(indexlist_remove)):
    
    combine = np.delete(combine, indexlist_remove[i], 0)
#-----------------------------------remove nan values ------------------------------



# ----------------------------------- calculate error -----------------------------
error = []
for i in range(combine.shape[0]):
    error.append(combine[i][2]-combine[i][4])
error = np.array(error)
error = error.reshape(error.shape[0],1)

combine = np.concatenate((combine,error),axis=1)
# ----------------------------------- calculate error -----------------------------


# ----------------------------------- calculate relative error -----------------------------
rel_error = []
for i in range(combine.shape[0]):
    rel_error.append(combine[i][6] / combine[i][4])
rel_error = np.array(rel_error)
rel_error = rel_error.reshape(rel_error.shape[0],1)

combine = np.concatenate((combine,rel_error),axis=1)
# ----------------------------------- calculate relative error -----------------------------



# ----------------------------------- Gender filtering -----------------------------
lenset = combine.shape[0]-1
for i in range(combine.shape[0]):
    ii = lenset - i
    if combine[ii][1] == 'none':
    #if combine[ii][1] == 'Female':
    #if combine[ii][1] == 'Male':                          # CHANGE MALE TO FEMALE TO SELECT GENDER
        combine = np.delete(combine,ii,0)
# ----------------------------------- Gender filtering -----------------------------
                                
                                
print(combine[0:6])
print(combine.shape)


# ----------------------------------- scatter points -----------------------------
fig, ax = plt.subplots()
ax.grid()
ax.set_axisbelow(True)
ax.scatter(combine[:,0], combine[:,6])
# ----------------------------------- scatter points -----------------------------


#-------------------------------------------- RANSAC FITTING ----------------------------------------------
X = combine[:,0].reshape(-1, 1)
y = combine[:,6].reshape(-1, 1)

ransac = RANSACRegressor(base_estimator=LinearRegression(),
                          min_samples=int(combine.shape[0]*0.5), max_trials=1000,
                          loss='absolute_loss', random_state=42)

ransac.fit(X, y)

line_X = np.arange(16, 30)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
#plt.plot(line_X, line_y_ransac, color='black', lw=2)
#-------------------------------------------- RANSAC FITTING ----------------------------------------------


#-------------------------------------------- LinearRegression ----------------------------------------------
X = combine[:,0].reshape(-1, 1)
y = combine[:,6]

model = LinearRegression()

model.fit(X, y)
# print('intercept:', model.intercept_)
# print('slope:', model.coef_)
line_X = np.arange(16, 30)
line_y_model = model.predict(line_X[:, np.newaxis])
plt.plot(line_X, line_y_model, color='black', lw=2)
#-------------------------------------------- LinearRegression ----------------------------------------------

#-------------------------------------------- t-Test ----------------------------------------------
# print("type y=",type(X[0]))
# print("type y=",type(X[0]))
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

#-------------------------------------------- t-Test ----------------------------------------------


ax.set_xlabel("BMI")
ax.set_ylabel("Estimation Error")
#ax.set_ylim([-1, 2])                #for combine[:,5]
ax.set_ylim([-10000, 10000])                #for combine[:,6]

plt.show()


#||||||||||||||||||||||||||||||||||||||||||||||Plot 1||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#||||||||||||||||||||||||||||||||||||||||||||||Plot 2||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# 1=16-18  2=18-20   3=20-22   4=22-24   5=24-26   6=26-32
section1 = []
section2 = []
section3 = []
section4 = []
section5 = []
section6 = []

for i in range(combine.shape[0]):
    if combine[i][0] < 18:
        section1.append(combine[i])
    if combine[i][0] >= 18 and combine[i][0] < 20:
        section2.append(combine[i])
    if combine[i][0] >= 20 and combine[i][0] < 22:
        section3.append(combine[i])
    if combine[i][0] >= 22 and combine[i][0] < 24:
        section4.append(combine[i])
    if combine[i][0] >= 24 and combine[i][0] < 26:
        section5.append(combine[i])
    if combine[i][0] >= 26:
        section6.append(combine[i])




section1 = np.array(section1)
section2 = np.array(section2)
section3 = np.array(section3)
section4 = np.array(section4)
section5 = np.array(section5)
section6 = np.array(section6)
#print(section2)

data = [section1[:,5]/section1[:,3],section2[:,5]/section2[:,3],section3[:,5]/section3[:,3],section4[:,5]/section4[:,3],section5[:,5]/section5[:,3],section6[:,5]/section6[:,3]] # uncomment for ipaqtot1 / total steps vs BMI
#data = [section1[:,3],section2[:,3],section3[:,3],section4[:,3],section5[:,3],section6[:,3]]   #total steps
#data = [section1[:,6],section2[:,6],section3[:,6],section4[:,6],section5[:,6],section6[:,6]]    #estimation error
#data = [section1[:,7],section2[:,7],section3[:,7],section4[:,7],section5[:,7],section6[:,7]]    # uncomment for relative estimation error

fig1, ax1 = plt.subplots()
ax1.grid(axis = 'y')

ax1.set_xlabel("BMI")
ax1.set_ylabel("Metabolic Equivalent of Task (MET) / Total Steps")
#ax1.set_ylabel("Metabolic Equivalent of Task (MET) / Total Steps")

#ax1.set_ylim([-8000, 8000])
#ax1.set_ylim([-1, 2])                   #uncomment this if you want the relative error vs BMI
ax1.set_ylim([-0.02, 0.2])             #uncomment this if you want the MET vs BMI

#-------labels
fig1.canvas.draw()
labels = [item.get_text() for item in ax1.get_xticklabels()]
labels[0] = '[0 - 18)'
labels[1] = '[18 - 20)'
labels[2] = '[20 - 22)'
labels[3] = '[22 - 24)'
labels[4] = '[24 - 26)'
labels[5] = '[26 - ∞)'
ax1.set_xticklabels(labels)



ax1.boxplot(data)

plt.show()

#||||||||||||||||||||||||||||||||||||||||||||||Plot 2||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#||||||||||||||||||||||||||||||||||||||||||||||Plot 3||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

data = [section1[:,7],section2[:,7],section3[:,7],section4[:,7],section5[:,7],section6[:,7]]    # uncomment for relative estimation error
fig2, ax2 = plt.subplots()
ax2.grid(axis = 'y')

ax2.set_xlabel("BMI")
ax2.set_ylabel("Relative Estimation Error")
ax2.set_ylim([-1, 2])                   #uncomment this if you want the relative error vs BMI

fig2.canvas.draw()
labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[0] = '[0 - 18)'
labels[1] = '[18 - 20)'
labels[2] = '[20 - 22)'
labels[3] = '[22 - 24)'
labels[4] = '[24 - 26)'
labels[5] = '[26 - ∞)'
ax2.set_xticklabels(labels)

ax2.boxplot(data)

plt.show()
