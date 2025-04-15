#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.what is parameter
#Parameter: In machine learning and statistics,
#a parameter is a numerical value that helps define a model's behavior.
#For example, in a linear regression model, the coefficients (like slope and intercept) are parameters
#that determine how the input variables influence the output.


# In[2]:


#what is Corelation?
#Correlation measures the relationship between two variables. 
#If two variables are positively correlated, an increase in one tends to lead to an increase in the other.
#If they are negatively correlated, an increase in one tends to lead to a decrease in the other.
#Correlation doesn't necessarily imply causation—it just shows a connection between variables.


# In[3]:


#what does negative corealation means?
#When two variables have a negative correlation, it means that as one increases, the other decreases.
#For example, in a business setting, if the price of a product increases,
#sales may decrease—that’s a negative correlation.


# In[4]:


#3.define machine learning what are the main components of machine learning?
#the subfield of computer science  that gives computers the ablity to learn without
#explicitly programmed.
#Machine learn patterns from the data and replicate the same in future.
#Main components of machine learning are-
#Features – Characteristics or attributes used to train the model, such as age, price, or temperature.

###Model – The mathematical structure that learns patterns from data and makes predictions.

#Training – The process where the model learns patterns from input data using algorithms.

#Testing & Validation – Checking the model’s performance on unseen data to ensure accuracy.

#Optimization & Evaluation – Techniques like hyperparameter tuning to refine model performance.

#Loss Function – Measures how far the model’s predictions are from actual values.


# In[5]:


#How does loss value help in determining whether the model is good or not?
#Loss value helps determine whether the model is performing well. 
#It quantifies the difference between actual values and predicted values:

#Low loss indicates the model makes accurate predictions.

#High loss suggests the model needs improvement (e.g., more data, better features, or different algorithms).

#Optimization techniques like gradient descent adjust model parameters to minimize loss.


# In[6]:


#What are continuous and categorical variables?
#Continuous Variables: These are numerical values that can take any number within a given range.
#They have a measurable quantity and can have decimals. 
#Examples include height, temperature, or stock prices.
#Categorical Variables: These represent distinct groups or categories. 
#They can be either nominal (no inherent order, like colors or city names)
#or ordinal (ordered categories, like star ratings or education level).


# In[7]:


#How do we handle categorical variables in Machine Learning? What are the common t echniques?
#Since machine learning models work with numbers, categorical variables need to be transformed into a numerical format.
#Common techniques include:

#Label Encoding: Assigns a unique integer to each category.
#Suitable for ordinal data (e.g., "Low," "Medium," "High" → 0, 1, 2).

#One-Hot Encoding: Converts categories into binary vectors 
#(e.g., "Red," "Blue," "Green" → [1,0,0], [0,1,0], [0,0,1]).
#Best for nominal categorical data.

#Target guided oridnal encoding
#based on relationship with target variable
#useful when we have large number of unique categories in categorical data.
#categorial groups with mean and median of corresponding target variable.




# In[8]:


#What do you mean by training and testing a dataset
#Training a Dataset: This is the process where a machine learning model learns from labeled data. 
#The model adjusts its parameters by analyzing patterns in the training dataset 
#so it can make accurate predictions.

#Testing a Dataset: After training, the model is tested on unseen data to evaluate its performance.
#The testing dataset helps determine if the model generalizes well to new information ,
#or if it has overfitted to the training data.


# In[9]:


#What is sklearn.preprocessing?
#sklearn.preprocessing is a module in Scikit-Learn, a popular Python library for machine learning.
#It provides various functions to scale, normalize, encode, and transform data before feeding it into a machine learning model.
#Some common preprocessing techniques include:
#Standardization (e.g., StandardScaler) – Adjusts features to have zero mean and unit variance.

#Normalization (e.g., MinMaxScaler) – Scales features to a specific range (e.g., 0 to 1).

#Encoding categorical variables (e.g., OneHotEncoder, LabelEncoder).

#Feature transformation (e.g., PolynomialFeatures, PowerTransformer)


# In[10]:


#What is a Test set
#A test set is a portion of the dataset that is not used for training but instead helps evaluate the model’s performance. 
#It ensures that the model generalizes well to unseen data. Typically, datasets are split into:

##Training set (e.g., 80%) – Used to train the model.

#Test set (e.g., 20%) – Used to assess accuracy and prevent overfitting. 


# In[11]:


# How do we split data for model fitting (training and testing) in Python?
# How do you approach a Machine Learning problem?
#In machine learning, we split data into training and testing sets to evaluate model performance. 
#The most common way to do this in Python is using train_test_split() from Scikit-Learn:
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample dataset
data = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5], 'Feature2': [10, 20, 30, 40, 50], 'Target': [0, 1, 0, 1, 0]})


X = data[['Feature1', 'Feature2']]  # Features
y = data['Target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)
#A structured approach helps ensure accuracy and efficiency in solving ML problems. 
#Here’s a step-by-step breakdown:

#Understand the Problem – Define the objective, whether it's classification, regression, clustering, etc.

#Collect & Explore Data – Perform Exploratory Data Analysis (EDA) to understand patterns, missing values, and distributions.

#Preprocess Data – Handle missing values, normalize/scale features, and encode categorical variables.

#Feature Engineering – Select or create meaningful features to improve model performance.

#Choose a Model – Select an appropriate algorithm (e.g., Decision Trees, Neural network.
#Train the Model – Fit the model using training data and optimize hyperparameters.

#Evaluate Performance – Use metrics like accuracy, precision, recall, RMSE, etc., to assess the model.

#Fine-Tune & Optimize – Adjust hyperparameters, try different models, or improve feature selection.

#Deploy & Monitor – Deploy the model and continuously monitor its performance in real-world scenarios.


# In[ ]:


#Why do we have to perform EDA before fitting a model to the data?
 #EDA is a crucial step in machine learning because it helps us understand the dataset before applying any model. 
    #Here’s why it’s important:

#Detects Missing Values & Outliers – Identifies anomalies that could negatively impact model performance.

#Feature Selection – Helps determine which variables are most relevant for prediction.

#Data Distribution Insights – Allows us to visualize how data is spread, ensuring appropriate transformations.

#Improves Model Accuracy – Cleaning and preprocessing data before training leads to better results.

#Prevents Overfitting – Helps avoid models learning noise instead of meaningful patterns.


# In[12]:


#What is Correlation?
#Correlation measures the relationship between two variables. 
#It tells us whether changes in one variable are associated with changes in another.



# In[13]:


#what does negative corealation means?
#When two variables have a negative correlation, it means that as one increases, the other decreases.
#For example, in a business setting, if the price of a product increases,
#sales may decrease—that’s a negative correlation.


# In[15]:


#How can you find correlation between variables in Python?
#In Python, you can calculate correlation using NumPy or Pandas.
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Calculate correlation coefficient
correlation_matrix = np.corrcoef(x, y)
print("Correlation Coefficient:", correlation_matrix[0, 1])


# In[16]:


import pandas as pd

# Sample DataFrame
data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})

# Compute correlation
print(data.corr())


# In[17]:


#What is causation? Explain difference between correlation and causation with an example
#Causation means that one event directly causes another.
#If variable X changes and directly leads to a change in variable Y, then X causes Y.

#Difference Between Correlation and Causation
#Correlation shows a relationship between two variables but does not imply causation.

#Causation means one variable directly influences another.

#Example
#Correlation: Ice cream sales and drowning incidents are positively correlated. 
#However, eating ice cream does not cause drowning.
#Instead, hot weather is a third factor influencing both.

#Causation: Smoking causes lung cancer because harmful chemicals damage lung cells.



# In[18]:


#What is an Optimizer? What are different types of optimizers?
#An optimizer is an algorithm used in machine learning and deep learning 
#to adjust model parameters (weights) to minimize the loss function.
#It helps the model learn by improving predictions over time.
#Gradient Descent
#Stochastic Gradient Descent (SGD)
#Adam (Adaptive Moment Estimation)
#RMSprop (Root Mean Square Propagation)


# In[19]:


#What is sklearn.linear_model?
#sklearn.linear_model is a module in Scikit-Learn that
#provides various linear models for regression and classification.
#Some key models include:

#Linear Regression (LinearRegression) – Fits a linear model using ordinary least squares.

##Ridge Regression (Ridge) – Adds L2 regularization to prevent overfitting.

#Lasso Regression (Lasso) – Uses L1 regularization for feature selection.

#Logistic Regression (LogisticRegression) – Used for binary classification problems.


# In[20]:


#What does model.fit() do? What arguments must be given?
#model.fit() is used to train a machine learning model
#by adjusting its parameters based on the given training data.
#It learns patterns from the input features (X_train) and their corresponding labels (y_train).

#Common Arguments for model.fit()
#X_train – The input features (training data).

#y_train – The target labels (for supervised learning).

#batch_size – Number of samples per gradient update (for deep learning models).

#epochs – Number of times the model iterates over the dataset.

#verbose – Controls the amount of output displayed during training
#(0 = silent, 1 = progress bar, 2 = one line per epoch).


# In[21]:


#What does model.predict() do? What arguments must be given?
#model.fit() is used to train a machine learning model
#by adjusting its parameters based on the given training data.
#It learns patterns from the input features (X_train) and their corresponding labels (y_train).

#Common Arguments for model.fit()
#X_train – The input features (training data).

#y_train – The target labels (for supervised learning).

#batch_size – Number of samples per gradient update (for deep learning models).

#epochs – Number of times the model iterates over the dataset.

#verbose – Controls the amount of output displayed during training 
#(0 = silent, 1 = progress bar, 2 = one line per epoch).


# In[22]:


#20.#What are continuous and categorical variables?
#Continuous Variables: These are numerical values that can take any number within a given range.
#They have a measurable quantity and can have decimals. 
#Examples include height, temperature, or stock prices.
#Categorical Variables: These represent distinct groups or categories. 
#They can be either nominal (no inherent order, like colors or city names)
#or ordinal (ordered categories, like star ratings or education level).


# In[23]:


#21. What is feature scaling? How does it help in Machine Learning?
 #Feature scaling is a preprocessing technique used in machine learning
    #to transform numerical features into a similar scale. 
    #It ensures that all features contribute equally to the model
    # preventing bias from features with larger values
    #Why is Feature Scaling Important?
#Improves Model Performance – Some algorithms (like gradient descent-based models)
#require features to be on the same scale for efficient learning.

#Prevents Bias – Features with larger values can dominate the learning process if not scaled properly.

#Enhances Convergence – Scaling helps optimization algorithms converge faster.

#Essential for Distance-Based Models – Algorithms like KNN and SVM rely on feature distances,
#making scaling crucial.


# In[27]:


#22.How do we perform scaling in Python?
#Python provides several methods for feature scaling using Scikit-Learn3:

#1. Standardization (Z-score normalization)
#Transforms data to have zero mean and unit variance.

from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[100, 200], [300, 400], [500, 600]])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)
#2. Min-Max Scaling
#Scales values between 0 and 1.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)
#3. Robust Scaling
#Handles outliers by scaling based on median and interquartile range.


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)


# In[28]:


#23.#What is sklearn.preprocessing?
#sklearn.preprocessing is a module in Scikit-Learn, a popular Python library for machine learning.
#It provides various functions to scale, normalize, encode, and transform data before feeding it into a machine learning model.
#Some common preprocessing techniques include:
#Standardization (e.g., StandardScaler) – Adjusts features to have zero mean and unit variance.

#Normalization (e.g., MinMaxScaler) – Scales features to a specific range (e.g., 0 to 1).

#Encoding categorical variables (e.g., OneHotEncoder, LabelEncoder).

#Feature transformation (e.g., PolynomialFeatures, PowerTransformer)


# In[29]:


# 24.How do we split data for model fitting (training and testing) in Python?
##In machine learning, we split data into training and testing sets to evaluate model performance. 
#The most common way to do this in Python is using train_test_split() from Scikit-Learn:
from sklearn.model_selection import train_test_split
import pandas as pd

# Sample dataset
data = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5], 'Feature2': [10, 20, 30, 40, 50], 'Target': [0, 1, 0, 1, 0]})


X = data[['Feature1', 'Feature2']]  # Features
y = data['Target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)
#A structured approach helps ensure accuracy and efficiency in solving ML problems. 
#Here’s a step-by-step breakdown:

#Understand the Problem – Define the objective, whether it's classification, regression, clustering, etc.

#Collect & Explore Data – Perform Exploratory Data Analysis (EDA) to understand patterns, missing values, and distributions.

#Preprocess Data – Handle missing values, normalize/scale features, and encode categorical variables.

#Feature Engineering – Select or create meaningful features to improve model performance.

#Choose a Model – Select an appropriate algorithm (e.g., Decision Trees, Neural network.
#Train the Model – Fit the model using training data and optimize hyperparameters.

#Evaluate Performance – Use metrics like accuracy, precision, recall, RMSE, etc., to assess the model.

#Fine-Tune & Optimize – Adjust hyperparameters, try different models, or improve feature selection.

#Deploy & Monitor – Deploy the model and continuously monitor its performance in real-world scenarios.


# In[ ]:


#25.Explain data encoding
#Converting the string into numerical data.
#Data encoding is the process of converting information into a specific format for efficient storage,
#transmission, or processing.
#It ensures that data can be understood by computers and transmitted securely across networks.

