# Linear Regression using Scikit-Learn
# It's an open-source, commercially usable machine learning toolkit [scikit-learn](https://scikit-learn.org/stable/index.html). 
# This toolkit contains implementations of many of the algorithms used for ML

# Utilize  scikit-learn to implement linear regression using Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from utils_common import  load_house_data

# Gradient Descent
# Scikit-learn has a gradient descent regression model [sklearn.linear_model.SGDRegressor]
# (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#examples-using-sklearn-linear-model-sgdregressor). 
#  [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) will perform z-score normalization. Here it is referred to as 'standard score'.

# Load data set

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# Scale/Normalize training data

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# Create and fit the regression model

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# View parameters
# The parameters are associated with the normalized input data

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# Make predictions
# Predict the targets of the training data. Use both the `predict` routine and compute using w and b.

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# Plot Results
# Plot the predictions versus the target values.

# plot predictions and targets vs original features    
# create a 1x4 grid of subplots
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

for i in range(len(ax)):  # since you created 4 subplots
    ax[i].scatter(X_train[:, i], y_train, label='target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], y_pred, color="orange", label='predict')

# y-label only on the first subplot
ax[0].set_ylabel("Price")
ax[0].legend()

# overall title
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

### Utilize  scikit-learn to implement linear regression using a close form solution based on the normal equation
# Linear Regression, closed-form solution
# Scikit-learn has the [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) which implements a closed-form linear regression.

''' Example 1: a house with 1000 square feet sold for $300,000 and a house with 2000 square feet sold for $500,000.

| Size (1000 sqft)     | Price (1000s of dollars) |
| ----------------| ------------------------ |
| 1               | 300                      |
| 2               | 500                      | '''

# Load the data set

X_train = np.array([1.0, 2.0])   #features
y_train = np.array([300, 500])   #target value

# Create and fit the model
'''The code below performs regression using scikit-learn. 
The first step creates a regression object.  
The second step utilizes one of the methods associated with the object, `fit`. 
This performs regression, fitting the parameters to the input data. 
The toolkit expects a two-dimensional X matrix. '''

linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train) 

# View parameters
# The w and b parameters are referred to as 'coefficients' and 'intercept' in scikit-learn.

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")

# Make Predictions

y_pred = linear_model.predict(X_train.reshape(-1, 1))

print("Prediction on training set:", y_pred)

X_test = np.array([[1200]])
print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test)[0]:0.2f}")

# Example 2: with multiple features.
# The closed-form solution work well on smaller data sets such as these but can be computationally demanding on larger data sets. 
# The closed-form solution does not require normalization.

# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

linear_model = LinearRegression()
linear_model.fit(X_train, y_train) 

b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")

print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")
