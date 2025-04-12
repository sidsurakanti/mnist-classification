# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %%
train = pd.read_csv("./dataset/train.csv")
test = pd.read_csv("./dataset/test.csv")

train = np.array(train)
m, n = train.shape # training examples (m), pixels (784)
np.random.shuffle(train)

train = train.T
X = train[1:] / 255.0 # images (784, m)
y = train[0] # labels (1, m)

test = np.array(test)
test = test.T
y_test, X_test = test[0], test[1:].T / 255.0

# %%
def params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def dReLU(Z):
    return (Z > 0).astype(float)

def softmax(Z2):
    expZ = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(y):
    m = y.shape[0]
    arr = np.zeros((10, m))
    arr[y, np.arange(m)] = 1
    return arr

# %%
def forward_pass(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    return A2, Z2, A1, Z1

# %%
def backward_pass(A2, Z2, A1, Z1, W2, W1, y):
    dZ2 = 2 * (A2 - one_hot(y))
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * dReLU(Z1)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2

def cost(A2, y):
    return (A2 - one_hot(y))**2 * 1/m

def update_params(dW1, db1, dW2, db2, W2, b2, W1, b1):
    lr = 0.1
    W2 -= dW2 * lr
    b2 -= db2 * lr
    W1 -= dW1 * lr
    b1 -= db1 * lr

    return W1, b1, W2, b1

def get_accuracy(A2, Y):
    predictions = np.argmax(A2, axis=0)
    return np.mean(predictions == y) * 100

def gradient_descent(X, Y, iterations):
    W1, b1, W2, b2 = params()
    for i in range(iterations):
        A2, Z2, A1, Z1 = forward_pass(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_pass(A2, Z2, A1, Z1, W2, W1, Y)
        W1, b1, W2, b2 = update_params(dW1, db1, dW2, db2, W2, b2, W1, b1)

        if i % 50 == 0:
            print("Iteration :", i)
            print("Accuracy :", round(get_accuracy(A2, Y), 2))
    return W1, b1, W2, b2

# %%
trainedWeights = gradient_descent(X, y, 500)

# %%
def predict(i, W1, b1, W2, b2):
    A2, _, _, _ = forward_pass(W1, b1, W2, b2, X_test.T)
    prediction = np.argmax(A2, axis=0)[i]
    label = y_test[i]
    plt.imshow(X_test[i].reshape((28, 28)), cmap="gray")
    plt.axis(False)
    print("Prediction: ", int(prediction))
    print("Label:", int(label))

predict(24, *trainedWeights)


