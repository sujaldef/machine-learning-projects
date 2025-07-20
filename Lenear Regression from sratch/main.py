import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('advertising.csv')
X = data[['TV', 'Radio', 'Newspaper']].values
y = data['Sales'].values
n = len(X)

# Hyperparameters
learning_rate = 0.00001
epochs = 10000

# Initialize weights
w = np.zeros(X.shape[1])  # [0, 0, 0]
b = 0

# Training
for epoch in range(epochs):
    y_pred = np.dot(X, w) + b
    dw = (-2/n) * np.dot(X.T, (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        loss = np.mean((y - y_pred)**2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {w}, b: {b:.4f}")

# Plot predicted vs actual
y_pred = np.dot(X, w) + b
plt.scatter(y, y_pred, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()
