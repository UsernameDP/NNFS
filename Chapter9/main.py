import numpy as np
import nnfs
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(np.array(inputs), self.weights) + self.biases


class Activation_RELU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities


class Loss:
    def calculate(self, y_pred, y_true):
        output_losses = self.forward(y_pred, y_true)

        return np.mean(output_losses)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # loop over each row, and find ele based on index in y_true
        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(len(y_pred_clipped)), y_true]
        # multiply the predictions by true values assuming row 1 of pred matches with row 1 of true
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(confidences)


class Accuracy:
    def calculate(self, y_pred, y_true):
        predictions = self.forward(y_pred, y_true)

        return np.mean(predictions)  # this is accuracy

    def forward(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)

        if (len(y_true.shape)) == 1:
            targets = y_true
        if (len(y_true.shape)) == 2:
            targets = np.argmax(y_true, axis=1)

        accuracies = predictions == targets

        return accuracies


nnfs.init()

coords, y_true = vertical_data(samples=100, classes=3)


x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b
a = max(z, 0)

# Derivative from next layer
dvalue = 1.0
drelu_dz = dvalue * (1.0 if z > 0 else 0.0)
print(drelu_dz)

# pg 186 continue
