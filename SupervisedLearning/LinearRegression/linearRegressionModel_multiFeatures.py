# Linear Regression model with multiple features
# Housing price prediction

# Extend Linear Regression model  routines to support multiple features
#    - Extend data structures to support multiple features
#    - Rewrite prediction, cost and gradient routines to support multiple features
#    - Utilize NumPy `np.dot` to vectorize their implementations for speed and simplicity

import copy, math
import numpy as np
import matplotlib.pyplot as plt

''' The training dataset contains three examples with four features (size, bedrooms, floors and, age. 
| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  '''

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# W is a vector with n elements.
#  - Each element contains the parameter associated with one feature.
#  - in this dataset, n is 4.
# b is a scalar parameter.  

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# The model's prediction with multiple variables is given by the linear model:
# fw,b(x) = w1*x1 + w2*x2 + ... + wn*xn + b
# where n is the number of features, w1, w2, ..., wn are the model parameters, 
# and b is the bias term.
# In vectorized form, this can be written as:
# fw,b(x) = w . x + b
# where w and x are n-dimensional vectors, and "." denotes the dot product.
'''
def predict_single_loop(x, w, b): 
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}") '''

def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# Cost function
# The cost function for linear regression with multiple variables is given by:
# J(w,b) = (1/2m) * Σ (fw,b(x(i)) - y(i))^2
# where m is the number of training examples, fw,b(x(i)) is the prediction for the i-th training example, 
# and y(i) is the actual output for the i-th training example.
def compute_cost(X, y, w, b ): 
    """
    Compute cost for linear regression with multiple variables
    
    Args:
      X (ndarray): Shape (m,n) training data, m examples with n features
      y (ndarray): Shape (m,) target values
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      total_cost (scalar):  cost
    """
    m = X.shape[0] # number of training examples
    total_cost = 0
    
    for i in range(m):
        f_wb = np.dot(X[i], w) + b   # prediction
        cost = (f_wb - y[i]) ** 2     # squared error
        total_cost = total_cost + cost # accumulate the total cost

    total_cost = total_cost / (2 * m) # average cost
    return total_cost

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

# Gradient Descent
# The gradients for linear regression with multiple variables are given by:
# ∂J(w,b)/∂wj = (1/m) * Σ (fw   ,b(x(i)) - y(i)) * xj(i) for j = 1, 2, ..., n
# ∂J(w,b)/∂b  = (1/m) * Σ (fw,b(x(i)) - y(i))
# where m is the number of training examples, fw,b(x(i)) is the prediction for the i-th training example, 
# y(i) is the actual output for the i-th training example, and xj(i) is the j-th feature of the i-th training example.

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# Gradient descent algorithm

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plot cost function history
plt.plot(J_hist)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs. Iteration")
plt.show()  

# Test the model with a custom input
# Predict the price of a 1650 sq-ft, 3 br, 1 floor, 40 year home
x_test = np.array([1650, 3, 1, 40])
price = predict(x_test, w_final, b_final)
print(f"Predicted price for a 1650 sq-ft, 3 br, 1 floor, 40 year home (using gradient descent): ${price*1000:0.2f}")    
    

