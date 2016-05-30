# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:27:14 2016

@author: Alex
"""

import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df_Train = pd.read_csv('C:/Users/Alex/Desktop/Data science training/Titanic/Data/train.csv', header=0)

describe = df_Train.describe()


df_Train['Gender']=0

df_Train.Gender[df_Train['Sex']=='male'] = 1
df_Train.Gender[df_Train['Sex']=='female'] = 0

#df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


median_ages = np.zeros((2,3))

for i in range(0,2):            #sex = 0 or 1
        for j in range(0,3):    #Pclass = 1, 2 or 3
            median_ages[i,j] = df_Train.Age[(df_Train['Gender']== i) & (df_Train['Pclass']== j+1)].dropna().median()


df_Train['AgeFill']= df_Train['Age'] #create a new col and copy Age

for i in range(0,2):            #sex = 0 or 1
        for j in range(0,3):    #Pclass = 1, 2 or 3
            df_Train.loc[ (df_Train.Age.isnull()) & (df_Train.Gender == i) & (df_Train.Pclass == j+1),'AgeFill'] = median_ages[i,j]


#add some columns
df_Train['AgeIsNull'] = pd.isnull(df_Train.Age).astype(int)

df_Train['FamilySize'] = df_Train['SibSp'] + df_Train['Parch']

df_Train['Age*Class'] = df_Train.AgeFill * df_Train.Pclass


#delete some columns not used
df_Train = df_Train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_Train = df_Train.drop(['Age'], axis=1)

df_Train = df_Train.dropna()
df_Train.head()
train_data = df_Train.values
train_data






#Test file
test_df = pd.read_csv('C:/Users/Alex/Desktop/Data science training/Titanic/Data/test.csv', header=0)    

ids = test_df.PassengerId

test_df['Gender']=0

test_df.Gender[test_df['Sex']=='male'] = 1
test_df.Gender[test_df['Sex']=='female'] = 0

median_ages = np.zeros((2,3))

for i in range(0,2):            #sex = 0 or 1
        for j in range(0,3):    #Pclass = 1, 2 or 3
            median_ages[i,j] = test_df.Age[(test_df['Gender']== i) & (test_df['Pclass']== j+1)].dropna().median()


test_df['AgeFill']= test_df['Age'] #create a new col and copy Age

for i in range(0,2):            #sex = 0 or 1
        for j in range(0,3):    #Pclass = 1, 2 or 3
            test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1),'AgeFill'] = median_ages[i,j]


#add some columns
test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass


#delete some columns not used
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.drop(['Age'], axis=1)

test_df = test_df.dropna()
test_df.head()





zip(ids, result_test)


from sklearn import tree
Y_Train = df_Train.Survived.values
X_Train = df_Train.drop("Survived",axis=1).values
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_Train, Y_Train)

X_test = test_df.values

result_test = clf.predict(X_test).astype(int)

import csv as csv
predictions_file = open("C:/Users/Alex/Desktop/Data science training/Titanic/decisiontree.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, result_test))
predictions_file.close()
print 'Done.'

