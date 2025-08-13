# Implement and Explore the cost function for linear regression with one variable
# Housing price prediction

import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
# dataset with only two data points - size of house (1000 sqft), price( 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# Computing Cost - cost is a measure of how well the model is predicting the target price of the house.
# Minimizing the cost can provide optimal values of w,b
# Because the difference between the target and pediction is squared in the cost equation, the cost increases rapidly 
# when w is either too large or too small.
# Using the `w` and `b` selected by minimizing cost results in a line which is a perfect fit to the data.

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
 
