# Linear Regression model f(w,b) with one variable
# Housing price prediction

# Linear regression builds a model which establishes a relationship between features and targets
#     - In this example, the feature is house size and the target is house price
#     - for simple linear regression, the model has two parameters w and b whose values are 'fit' using training data.
#     - once a model's parameters have been determined, the model can be used to make predictions on novel data.

import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
# dataset with only two data points - size of house (1000 sqft), price( 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
# Or use len(x_train)
print(f"Number of training examples is: {m}")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

# Model function
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    print(f'f_wb = {f_wb}')
    for i in range(m):
        f_wb[i] = w * x_train[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# Prediction
# Now that we have a model, use it to make prediction. Let's predict the price of a house with 1200 sqft. 
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")
