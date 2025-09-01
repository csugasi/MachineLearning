''' Plot the decision boundary for a logistic regression model, which will give a better sense of what the model is predicting.

## Dataset
Example:
- The input variable `X` is a numpy array which has 6 training examples, each with two features
- The output variable `y` is also a numpy array with 6 examples, and `y` is either `0` or `1` '''

import numpy as np
from utils_common import plot_data, sigmoid
import matplotlib.pyplot as plt


X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

# Plot data
# Data points with label y=1 are red crosses, data points with label y=0 are black circles

fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')

ax.set_xlabel('$x_0$')
plt.show()

# Logistic Regression model 
''' For logistic regression, the model is represented as 

  fw,b(x^i) = g(w.x^i+b)

  where

  g(z) = 1/(1+e^(-z))
  
  g(z) is known as the sigmoid function and it maps all input values to values between 0 and 1.

  Ex: Lets say we trained the model and got the params as b=-3, w0=1, w1=1, that is:
  f(x) = g(x0+x1-3)
  

  We interpret the output of the model f(x) as the probability that y=1 given x and parameterized by w.
  Therefore, to get a final prediction (y=0 or y=1) from the logistic regression model, use the following heuristic -

  if f(x) >= 0.5, predict y=1
  
  if f(x) < 0.5, predict y=0
  
  Since, f(x) = g(w^Tx), let's plot the sigmoid function to see where g(z) >= 0.5 '''

# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)

''' From plot, we can see g(z) >= 0.5 for z >= 0
For logistic regression model z = w.x+b, 
If w.x+b >= 0, the model predicts y = 1
If w.x+b < 0, the model predicts y = 0

# Plotting decision boundary
''' Logistic regression model has the form f(x) = g(-3 + x0 + x1)
This model predicts y = 1 if = -3 + x0 + x1 >= 0. ( x1 >= 3 - x0) '''

# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()

# In plot, the blue line represents the line x0 + x1 -3 = 0. Any point in the shaded region ( under the line)
# is classified as y = 0. Any point on or above the line is classified as y = 1. 
# This line is known as the decision boundary
# By using higher order polynomial terms Ex: f(x) = g( x0^2 + x1 -1), we can come up with more complex non-linear boundaries.




