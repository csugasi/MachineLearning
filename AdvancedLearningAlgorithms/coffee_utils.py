import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.activations import sigmoid

def load_coffee_data():
    """Creates a coffee roasting data set.
       Roasting duration: 12–15 minutes is best
       Temperature range: 175–260 °C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5           # 12–15 min is best
    X[:,0] = X[:,0] * (285-150) + 150    # 175–260 C is best
    Y = np.zeros(len(X))
    
    for i, (t,d) in enumerate(X):
        y = -3/(260-175)*t + 21
        if (175 < t < 260) and (12 < d < 15) and (d <= y):
            Y[i] = 1
    return X, Y.reshape(-1,1)

def plt_roast(X, Y):
    Y = Y.reshape(-1,)
    fig, ax = plt.subplots(figsize=(6,5))
    
    # Good roasts (Y=1)
    ax.scatter(X[Y==1,0], X[Y==1,1],
               s=70, marker='x', c='red', label="Good Roast")
    
    # Bad roasts (Y=0)
    ax.scatter(X[Y==0,0], X[Y==0,1],
               s=100, marker='o', facecolors='none',
               edgecolors='blue', linewidth=1.2, label="Bad Roast")
    
    # Decision boundary + guidelines
    tr = np.linspace(175,260,50)
    ax.plot(tr, (-3/85) * tr + 21, color='purple', linewidth=1.2)
    ax.axhline(y=12, color='purple', linewidth=1)
    ax.axvline(x=175, color='purple', linewidth=1)
    
    ax.set_title("Coffee Roasting", fontsize=16)
    ax.set_xlabel("Temperature (Celsius)", fontsize=12)
    ax.set_ylabel("Duration (minutes)", fontsize=12)
    ax.legend(loc='upper right')
    plt.show()
