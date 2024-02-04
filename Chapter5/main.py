import numpy as np
import nnfs
from nnfs.datasets import spiral_data
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

Coords, Classes = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)
activation1 = Activation_RELU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
accuracy_function = Accuracy()

dense1.forward(Coords)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)

outputs = np.array(activation2.outputs)


print(loss_function.calculate(outputs, Classes))
print("Acc : ", accuracy_function.calculate(outputs, Classes))
