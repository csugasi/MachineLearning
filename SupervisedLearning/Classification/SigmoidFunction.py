''' Implement the sigmoid function (logistic function)

 For a classification task, we can start by using linear regression model (f(x) = w^Tx) to predict `y` given `X`. 
- However, we'd like the predictions of classification model to be between 0 and 1, since output variable `y` is either 0 or 1. 
- This can be accomplished by using a "sigmoid function", which maps all input values to values between 0 and 1. 

## Formula for Sigmoid function

The formula for a sigmoid function is as follows -  

$g(z) = 1/{1+e^{-z}}

In the case of logistic regression, `z` (the input to the sigmoid function), is the output of a linear regression model. 
- That is, `z` is not always a single number, but can also be an array of numbers. 
- If the input is an array of numbers, apply the sigmoid function to each value in the input array. '''

import numpy as np
import matplotlib.pyplot as plt

# Input is an array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

def sigmoid(z):
  
  g = 1 / (1 + np.exp(-z))
  
  return g

# Output of sigmoid function for various value of `z`
# Generate an array of evenly spaced values between -10 and 10
z = np.arange(-10,10)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Ouput (sigmoid(z))")
print(np.c_[z, y])

# Plot z vs sigmoid(z)
plt.plot(z, y, c="b")

# Set the title
plt.title("Sigmoid function")
# Set the y-axis label
plt.ylabel('sigmoid(z)')
# Set the x-axis label
plt.xlabel('z')


