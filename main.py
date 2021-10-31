We will perform this entire task in 7 steps that includes:
1.EXPLORATORY DATA ANALYSIS(EDA)
2.DATA VISUALIZATION
3.PREPARING THE DATA
4.SPLITTING THE TRAINING AND TESTING DATA
5.MODEL CREATION AND TRAINING
6.COMPARING ACTUAL AND PREDICTED VALUES
7.MODEL EVALUATION
1. Exploratory Data Analysis(EDA)
In [1]:
# importing all the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plp
import seaborn as sns
In [2]:
# Reading the given data set using pandas library
df = pd.read_csv("http://bit.ly/w-data")
In [3]:
# Display all column names
df.columns
Out[3]:
Index(['Hours', 'Scores'], dtype='object')
In [4]:
# Display first five lines of data
df.head()

In [5]:
df.size
Out[5]:
50
In [6]:
df.shape # it prints the total no. of columns and total no. of rows
Out[6]:
(25, 2)
In [7]:
#Finding number of null values or missing values in the given data
df.isnull().sum()
Out[7]:
Hours     0
Scores    0
dtype: int64
In [8]:
# Average of hours that a student study
df['Hours'].mean()
Out[8]:
5.012
In [9]:
#Describe the given data
df.describe()

In [10]:
#more information about data
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
2. Data Visualization
By using data visualization we can understand more about data, like relation between two features.
In [11]:
# taking Hours column in X-axis
x = df['Hours']
# taking Scores column in Y-axis
y = df['Scores']
In [12]:
# finding the relationship between hours and scores of student by using scatter plot.
plp.scatter(x,y)
Out[12]:
<matplotlib.collections.PathCollection at 0xb516948>

In [13]:
# finding the relationship between hours and scores of student by using regression plot.
sns.regplot(x,y)
Out[13]:
<matplotlib.axes._subplots.AxesSubplot at 0xbe8dc88>

In [14]:
# finding the speread of  Hours data using histogram plot.
sns.distplot(x)
Out[14]:
<matplotlib.axes._subplots.AxesSubplot at 0xbf18948>

In [15]:
# finding the speread of Scores data using histogram plot.
sns.distplot(y)
Out[15]:
<matplotlib.axes._subplots.AxesSubplot at 0xbf917c8>

In [16]:
plp.bar(x,y)
Out[16]:
<BarContainer object of 25 artists>




3. Preparing The Data
In [17]:
# Taking x as an independent variable and it must be in 2Dimentional-array
x=df[['Hours']]

# Taking y as a dependent variable and it must be in 1Dimentional-array
y=df['Scores']
In [18]:
x.head()
Out[18]:

In [19]:
y.head()
Out[19]:
0    21
1    47
2    27
3    75
4    30
Name: Scores, dtype: int64
In [20]:
x.shape
Out[20]:
(25, 1)
In [21]:
y.shape
Out[21]:
(25,)

4. Spliting The Training And Testing Data
We split the given data into train data for training the machine learning model and test data for testing that model.
In [22]:
#importing the necessary module for splitiing the data
from sklearn.model_selection import train_test_split
In [117]:
# Spliting the data.
X_train,x_test,Y_train,y_test=train_test_split(x,y)
In [118]:
X_train.shape
Out[118]:
(18, 1)
In [119]:
Y_train.shape
Out[119]:
(18,)
In [120]:
x_test.shape
Out[120]:
(7, 1)
In [121]:
y_test.shape
Out[121]:
(7,)
In [122]:
# Now the data is ready to create the model
5. Model Creation & Training
Here we will implement a linear regression model by using the train data.
In [123]:
#importing the model
from sklearn.linear_model import LinearRegression
In [124]:
#Creating the Model Object
model = LinearRegression()
In [125]:
# Training the model
model.fit(X_train,Y_train)
print("Model Training is Completed.")
Model Training is Completed.
In [126]:
# regression plot for training data.
sns.regplot(X_train,Y_train)
Out[126]:
<matplotlib.axes._subplots.AxesSubplot at 0xcc1ea48>

In [128]:
# predicting values for test data
predicted = model.predict(x_test)
In [138]:
predicted
Out[138]:
array([28.75648026, 26.82265248, 77.10217468, 51.96241358, 21.02116915,
       61.63155247, 36.49179136])
6. Comparing Actual and Predicted Values
In [129]:
# creating a data frame for y_test and predicted data.
diff = pd.DataFrame({"Actual":y_test,"Predicted":predicted})
In [137]:
diff
Out[137]:

In [136]:
# Comparing the actual and predicted values using bar graphs.
diff.plot(kind='bar')
Out[136]:
<matplotlib.axes._subplots.AxesSubplot at 0xdd9b208>


7. Model Evaluation
In [142]:
# importing the necessary metrics for modal evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
In [143]:
print(mean_absolute_error(y_test,predicted))
4.588596558948565
In [144]:
print(mean_squared_error(y_test,predicted))
25.497155599755526
In [145]:
print(r2_score(y_test,predicted))
0.9442846671250437
Given Problem Statement :
What will be predicted score if a student studies for 9.25 hrs/ day ?
In [147]:
hours = [[9.25]]

# making prediction
pred_score = model.predict(hours)

print("Hours Studied : {} \nScore Predicted : {}".format(hours,pred_score))
Hours Studied : [[9.25]] 
Score Predicted : [92.08933995]
