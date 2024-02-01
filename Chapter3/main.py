import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(np.array(inputs), self.weights) + self.biases


nnfs.init()

inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2], [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
weights2 = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(np.array(inputs), np.array(weights).T) + biases
layer2_outputs = np.dot(np.array(layer1_outputs), np.array(weights2)) + biases2

print(layer2_outputs)

Coord2D, Category = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
dense1.forward(Coord2D[:5])

print(dense1.output)

plt.scatter(Coord2D[:, 0], Coord2D[:, 1], c=Category, cmap="brg")
plt.show()
