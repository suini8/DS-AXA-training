# -*- coding: utf-8 -*-
"""
Created on Fri May 20 21:40:19 2016

@author: Alex
"""
'''
# The first thing to do is to import the relevant packages
# that I will need for my script, 
# these include the Numpy (for maths and arrays)
# and csv for reading and writing csv files
# If i want to use something from this I need to call 
# csv.[function] or np.[function] first

import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('C:/Users/Alex/Desktop/Data science training/Titanic/Data/train.csv', 'rb')) 
header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format




print data[0]
print data[0][0]
print data[0,3]
print data[0::,4] # all the column 4 (0:: - from 0 to end) 


data[0::,2].astype(np.float) #transform data type into float


# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[0::,4] == "female" # This finds where all 
                                           # the elements in the gender
                                           # column that equals “female”
men_only_stats = data[0::,4] != "female"   # This finds where all the 
                                           # elements do not equal 
                                           # female (i.e. male)



# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 

# and then print it out
print 'Proportion of women who survived is ' + str(round(proportion_women_survived , 2))
print 'Proportion of men who survived is ' + str( round(proportion_men_survived , 2))






'''
-------------------------------------------------------------
'''
# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

# I know there were 1st, 2nd and 3rd classes on board
number_of_classes = 3

# But it's better practice to calculate this from the data directly
# Take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[0::,2])) 

# Initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))



for i in xrange(number_of_classes):       #loop through each class
  for j in xrange(number_of_price_brackets):   #loop through each price bin
      women_only_stats = data[(data[0::,4] == "female") &(data[0::,2].astype(np.float) == i+1)&(data[0:,9].astype(np.float) >= j*fare_bracket_size)&(data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1]                                                  
      men_only_stats = data[(data[0::,4] != "female")&(data[0::,2].astype(np.float) == i+1)&(data[0:,9].astype(np.float)>= j*fare_bracket_size)&(data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1] 
      
      survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) 
      survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

survival_table[ survival_table != survival_table ] = 0.






"""
=========================================================================================================================

"""


# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:27:14 2016

@author: Alex
"""

import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('C:/Users/Alex/Desktop/Data science training/Titanic/Data/train.csv', header=0)
df.dtypes

df.describe()

df.Age[0:10].describe()

test = df[(df['Age'] == 60) & (df['Pclass']==1)]


test2= df.Age[(df['Pclass']==1)]
test3= df.Age[(df['Pclass']==2)]
test4= df.Age[(df['Pclass']==3)]

test2.describe()
test3.describe()
test4.describe()

df['Age'].isnull()


import pylab as P
df['Age'].hist()
P.show()


import pylab as P
df.Age[df['Survived'] == 1].hist()
P.show()

import pylab as P
df.Age[df['Survived'] == 0].hist()
P.show()

df.head()

df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()

df['Gender']=0

df.Gender[df['Sex']=='male'] = 1
df.Gender[df['Sex']=='female'] = 0

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


median_ages = np.zeros((2,3))

for i in range(0,2):            #sex = 0 or 1
        for j in range(0,3):    #Pclass = 1, 2 or 3
            #median_ages[i,j] = df.Age[(df['Gender']== i) & (df['Pclass']== j)].median() # they used drop.na
            median_ages[i,j] = df.Age[(df['Gender']== i) & (df['Pclass']== j+1)].dropna().median()


df['AgeFill']= df['Age'] #create a new col and copy Age

for i in range(0,2):            #sex = 0 or 1
        for j in range(0,3):    #Pclass = 1, 2 or 3
            #df[df['Age'].isnull()]
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]
            #df[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]

#add some columns
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass


#delete some columns not used
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.drop(['Age'], axis=1)

df = df.dropna()

train_data = df.values
train_data


df.dtypes




#write csv
test_file = open('C:/Users/Alex/Desktop/Data science training/Titanic/Data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
predictions_file = open("'C:/Users/Alex/Desktop/Data science training/Titanic/Data/genderclassmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])


'''

#################################################################################################

""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""


import csv as csv
import numpy as np

csv_file_object = csv.reader(open('C:/Users/Alex/Desktop/Data science training/Titanic/Data/train.csv', 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
data=[] 												# Create a variable to hold the data

for row in csv_file_object: 							# Skip through each row in the csv file,
    data.append(row[0:]) 								# adding each row to the data variable
data = np.array(data) 									# Then convert from a list to an array.

# Now I have an array of 12 columns and 891 rows
# I can access any element I want, so the entire first column would
# be data[0::,0].astype(np.float) -- This means all of the rows (from start to end), in column 0
# I have to add the .astype() command, because
# when appending the rows, python thought it was a string - so needed to convert

# Set some variables
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers 

# I can now find the stats of all the women on board,
# by making an array that lists True/False whether each row is female
women_only_stats = data[0::,4] == "female" 	# This finds where all the women are
men_only_stats = data[0::,4] != "female" 	# This finds where all the men are (note != means 'not equal')

# I can now filter the whole data, to find statistics for just women, by just placing
# women_only_stats as a "mask" on my full data -- Use it in place of the '0::' part of the array index. 
# You can test it by placing it there, and requesting column index [4], and the output should all read 'female'
# e.g. try typing this:   data[women_only_stats,4]
women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

# and derive some statistics about them
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

# Now that I have my indicator that women were much more likely to survive,
# I am done with the training set.
# Now I will read in the test file and write out my simplistic prediction:
# if female, then model that she survived (1) 
# if male, then model that he did not survive (0)

# First, read in test.csv
test_file = open('C:/Users/Alex/Desktop/Data science training/Titanic/Data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Also open the a new file so I can write to it. Call it something descriptive
# Finally, loop through each row in the train file, and look in column index [3] (which is 'Sex')
# Write out the PassengerId, and my prediction.

predictions_file = open("gendermodel.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])	# write the column headers
for row in test_file_object:									# For each row in test file,
    if row[3] == 'female':										# is it a female, if yes then
        predictions_file_object.writerow([row[0], "1"])			# write the PassengerId, and predict 1
    else:														# or else if male,
        predictions_file_object.writerow([row[0], "0"])			# write the PassengerId, and predict 0.
test_file.close()												# Close out the files.
predictions_file.close()

#copy in same folder as code


########################################

""" Now that the user can read in a file this creates a model which uses the price, class and gender
Author : AstroDave
Date : 18th September 2012
Revised : 28 March 2014

"""


import csv as csv
import numpy as np

csv_file_object = csv.reader(open('C:/Users/Alex/Desktop/Data science training/Titanic/Data/train.csv', 'rb')) 	# Load in the csv file
header = csv_file_object.next()                             # Skip the fist line as it is a header
data=[]                                                     # Create a variable to hold the data

for row in csv_file_object:                 # Skip through each row in the csv file
    data.append(row)                        # adding each row to the data variable
data = np.array(data)                       # Then convert from a list to an array

# In order to analyse the price column I need to bin up that data
# here are my binning parameters, the problem we face is some of the fares are very large
# So we can either have a lot of bins with nothing in them or we can just lose some
# information by just considering that anythng over 39 is simply in the last bin.
# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size
number_of_classes = 3                             # I know there were 1st, 2nd and 3rd classes on board.
number_of_classes = len(np.unique(data[0::,2]))   # But it's better practice to calculate this from the Pclass directly:
                                                  # just take the length of an array of UNIQUE values in column index 2


# This reference matrix will show the proportion of survivors as a sorted table of
# gender, class and ticket fare.
# First initialize it with all zeros
survival_table = np.zeros([2,number_of_classes,number_of_price_brackets],float)

# I can now find the stats of all the women and men on board
for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):

        women_only_stats = data[ (data[0::,4] == "female") \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                                 & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1]

        men_only_stats = data[ (data[0::,4] != "female") \
                                 & (data[0::,2].astype(np.float) == i+1) \
                                 & (data[0:,9].astype(np.float) >= j*fare_bracket_size) \
                                 & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1]

                                 #if i == 0 and j == 3:

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))  # Female stats
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))    # Male stats

# Since in python if it tries to find the mean of an array with nothing in it
# (such that the denominator is 0), then it returns nan, we can convert these to 0
# by just saying where does the array not equal the array, and set these to 0.
survival_table[ survival_table != survival_table ] = 0.

# Now I have my proportion of survivors, simply round them such that if <0.5
# I predict they dont surivive, and if >= 0.5 they do
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1

# Now I have my indicator I can read in the test file and write out
# if a women then survived(1) if a man then did not survived (0)
# First read in test
test_file = open('C:/Users/Alex/Desktop/Data science training/Titanic/Data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

# Also open the a new file so I can write to it. 
predictions_file = open("genderclassmodel.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])

# First thing to do is bin up the price file
for row in test_file_object:
    for j in xrange(number_of_price_brackets):
        # If there is no fare then place the price of the ticket according to class
        try:
            row[8] = float(row[8])    # No fare recorded will come up as a string so
                                      # try to make it a float
        except:                       # If fails then just bin the fare according to the class
            bin_fare = 3 - float(row[1])
            break                     # Break from the loop and move to the next row
        if row[8] > fare_ceiling:     # Otherwise now test to see if it is higher
                                      # than the fare ceiling we set earlier
            bin_fare = number_of_price_brackets - 1
            break                     # And then break to the next row

        if row[8] >= j*fare_bracket_size\
            and row[8] < (j+1)*fare_bracket_size:     # If passed these tests then loop through
                                                      # each bin until you find the right one
                                                      # append it to the bin_fare
                                                      # and move to the next loop
            bin_fare = j
            break
        # Now I have the binned fare, passenger class, and whether female or male, we can
        # just cross ref their details with our survival table
    if row[3] == 'female':
        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare ])])
    else:
        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare])])

# Close out the files
test_file.close()
predictions_file.close()

###################################################################################

""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('C:/Users/Alex/Desktop/Data science training/Titanic/Data/train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# TEST DATA
test_df = pd.read_csv('C:/Users/Alex/Desktop/Data science training/Titanic/Data/test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("C:/Users/Alex/Desktop/Data science training/Titanic/myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

