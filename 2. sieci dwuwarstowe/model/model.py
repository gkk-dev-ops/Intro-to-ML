import numpy as np
import matplotlib.pyplot as plt


def init2(S, K1, K2):
    W1 = np.random.rand(S + 1, K1) * 0.2 - 0.1
    W2 = np.random.rand(K1 + 1, K2) * 0.2 - 0.1
    return W1, W2


def dzialaj2(W1, W2, X):
    beta = 5
    # Ensure X is a column vector. Reshape if necessary.
    if X.ndim == 1:
        # Reshape X to be a column vector if it's a 1D array
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        raise ValueError("Input X must be a 1D array or a 2D column vector.")

    # Adding the bias term correctly
    # np.vstack expects that all input arrays have the same number of columns
    X1 = np.vstack((-1, X))
    U1 = np.dot(W1.T, X1)
    Y1 = 1 / (1 + np.exp(-beta * U1))
    X2 = np.vstack((-1, Y1))
    U2 = np.dot(W2.T, X2)
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2


def ucz2(W1, W2, P, T, n):
    liczbaPrzykladow = P.shape[1]
    wspUcz = 0.1
    beta = 5
    errors = []

    for i in range(n):
        nrPrzykladu = np.random.randint(liczbaPrzykladow)
        X = P[:, nrPrzykladu]
        Y1, Y2 = dzialaj2(W1, W2, X)
        Tn = T[nrPrzykladu]

        D2 = Tn - Y2
        E2 = beta * D2 * Y2 * (1 - Y2)
        D1 = W2[1:] * E2  # ignore bias weight in backpropagation
        E1 = beta * D1 * Y1 * (1 - Y1)

        X1 = np.vstack((-1, X))
        X2 = np.vstack((-1, Y1))

        dW1 = wspUcz * np.outer(X1, E1)
        dW2 = wspUcz * np.outer(X2, E2)

        W1 += dW1
        W2 += dW2

        error = np.sum(0.5 * (Tn - Y2) ** 2)
        errors.append(error)

    return W1, W2, errors


def plot_errors(errors):
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Training Iterations')
    plt.show()
