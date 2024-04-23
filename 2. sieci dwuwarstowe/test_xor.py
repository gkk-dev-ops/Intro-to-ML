import numpy as np
from model.model import init2, plot_errors, ucz2, dzialaj2

P = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
T = np.array([0, 1, 1, 0])
W1, W2 = init2(2, 2, 1)

W1, W2, errors = ucz2(W1, W2, P, T, 5000)
plot_errors(errors)

# Testing trained network
results = []
for i in range(P.shape[1]):
    # Make sure to pass each column as a column vector
    X = P[:, i:i+1]  # Slicing to keep the second dimension
    _, Y2 = dzialaj2(W1, W2, X)
    results.append(Y2.flatten())  # Flatten Y2 to convert it from 2D to 1D

print("Results after training:", results)
