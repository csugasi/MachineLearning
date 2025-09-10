# Train a logistic regression model using scikit-learn
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# Fit the model 
# Fit the logistic regression model on the training data by calling fit function

lr_model = LogisticRegression()
lr_model.fit(X, y)

# Make predictions
# Predictions made by the model can be seen by calling the predict function

y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)

# Calculate accuracy
# Calculate accuracy of the model by called score function

print("Accuracy on training set:", lr_model.score(X, y))
