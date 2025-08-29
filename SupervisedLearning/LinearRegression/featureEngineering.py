'''Linear regression can model complex, even highly non-linear functions using feature engineering
- It is important to apply feature scaling when doing feature engineering '''

import numpy as np
import matplotlib.pyplot as plt
from utils_common import zscore_normalize_features, run_gradient_descent_feng

# Polynomial Features

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()


# Engineer features 
X = x**2      #<-- added engineered feature

X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# Selecting Feature
# It may not always be obvious which features are required. We can add a variety of potential features to try and find the most useful. 
# For example,  y=w_0x_0 + w_1x_1^2 + w_2x_2^3+b 

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature

model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

''' Note the value of w is `[0.08 0.54 0.03]` and b is `0.0106`.This implies the model after fitting/training is: 0.08x + 0.54x^2 + 0.03x^3 + 0.0106 
Gradient descent has emphasized the data that is the best fit to the x^2 data by increasing the w_1 term relative to the others.  
If you were to run for a very long time, it would continue to reduce the impact of the other terms. 
Gradient descent is picking the 'correct' features for by emphasizing its associated parameter

- less weight value implies less important/correct feature, and in extreme, when the weight becomes zero or very close to zero, the associated feature is not useful in fitting the model to the data.
- In above example, after fitting, the weight associated with the x^2 feature is much larger than the weights for x or x^3 as it is the most useful in fitting the data. '''


# Scaling features

# If the data set has features with significantly different scales, we should apply feature scaling to speed gradient descent. 
# In the example above, there is x, x^2 and x^3 which will have very different scales. 
# Applying Z-score normalization to other above example

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# Complex functions
# With feature engineering, even quite complex functions can be modeled
# It is important to apply feature scaling when doing feature engineering

y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

