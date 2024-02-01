import numpy as np
import nnfs

nnfs.init()

a = [1, 2, 3]
b = [2, 3, 4]
dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
print(dot_product)

inputs = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
layer_outputs = np.dot(np.array(inputs), np.array(weights).T) + biases
print(layer_outputs)
