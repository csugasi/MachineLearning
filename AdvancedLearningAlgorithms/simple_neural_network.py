# Build a small neural network using tensorflow
''' Example: Coffee Roasting - The two features are Temperature in Celsius and Duration in minutes. Coffee Roasting at home suggests that the duration is best kept between 12 and 15 minutes
 while the temp should be between 175 and 260 degrees Celsius. 
 Of course, as temperature rises, the duration should shrink. '''

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# DataSet

X,Y = load_coffee_data();
print(X.shape, Y.shape)

plt_roast(X,Y)

# Normalize data

''' Fitting the weights to the data (back-propagation) will proceed more quickly if the data is normalized. 

The procedure below uses a Keras [normalization layer]. It has the following steps:
- create a "Normalization Layer". Note, as applied here, this is not a layer in the model.
- 'adapt' the data. This learns the mean and variance of the data set and saves the values internally.
- normalize the data.  
It is important to apply normalization to any future data that utilizes the learned model.'''

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

# Tile/copy data to increase the training set size and reduce the number of training epochs.

Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)   

# Tensorflow model - two layers with sigmoid activations 

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)

# Note 1:  The `tf.keras.Input(shape=(2,)),` specifies the expected shape of the input. 
# This allows Tensorflow to size the weights and bias parameters at this point. 
# This is useful when exploring Tensorflow models. 
# Above statement can be omitted in practice and Tensorflow will size the network parameters 
# when the input data is specified in the `model.fit` statement.  

# Note 2: Including the sigmoid activation in the final layer is not considered best practice. 
# It would instead be accounted for in the loss which improves numerical stability. 

model.summary()

L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

# Examine the weights and biases Tensorflow has instantiated.  The weights W should be of size (number of features in input, 
# number of units in the layer) while the bias b size should match the number of units in the layer:
# In the first layer with 3 units, we expect W to have a size of (2,3) and b should have 3 elements.
# In the second layer with 1 unit, we expect W to have a size of (3,1) and b should have 1 element.

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
) # The `model.compile` statement defines a loss function and specifies a compile optimization.

model.fit(
    Xt,Yt,            
    epochs=10,
) # The `model.fit` statement runs gradient descent and fits the weights to the data.

# Epochs and batches
'''In the `fit` statement above, the number of `epochs` was set to 10. This specifies that the entire data set should be applied during training 10 times. 
 During training, we see output describing the progress of training that looks like this:
```
Epoch 1/10
6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
```
The first line, `Epoch 1/10`, describes which epoch the model is currently running. 
For efficiency, the training data set is broken into 'batches'. 
The default size of a batch in Tensorflow is 32. 
There are 200000 examples in this expanded data set or 6250 batches. 
The notation on the 2nd line `6250/6250 [====` is describing which batch has been executed.'''

# Updated weights: After fitting, the weights have been updated.
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# Predictions
''' Once you have a trained model, we can then use it to make predictions. 
The output of the model is a probability. In this case, the probability of a good roast. 
To make a decision, one must apply the probability to a threshold. In this case, we will use 0.5 

Create input data. The model is expecting one or more examples where examples are in the rows of matrix.
 In this case, we have two features so the matrix will be (m,2) where m is the number of examples.
Since we normalized the input features, we must normalize test data as well.   
To make a prediction, apply the `predict` method.'''

X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

# To convert the probabilitites to a decision, we apply a threshold

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

# yhat = (predictions >= 0.5).astype(int) 
