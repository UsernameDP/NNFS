# Pip Libraries

- `numpy`
- `nnfs`
- `matplotlib`
- `PyQt5`

## Tips

- When you do matrix multiplication, the input and output have the same structure (input[0][0] maps to output[0][0])
- In a weight matrix (assuming transposed), the row represents the number of inputs and the columns represent the number of neurons.

## Vocab

- `Inputs` - The very first 'layer' of the neural networks.
- `Neurons` - The neuron has a weight for each input from the previous layer and has a bias.
- `Layer` - Composed of Neurons or inputs.
- `Layer Outputs` - The number of layer outputs is equal to the number of neurons in that layer.
- `Outputs` - The final outputs of the entire network.

### Activation Functions

- The activation function is applied after weights AND bias.
- Activation function maps inputs to a value on the function. Hence, the outputs of the actiation function are limited by the definition of the activation function.
- In essense, activation function is the function that is recursively used each `Layer` and can be used to variably map specific
  `Inputs` to specific `Outputs`.

#### RELU Activation Function

$$
\text{f}(x) =
\begin{cases}
0 & \text{if } x \leq 0,\\
x & \text{if } x > 0.
\end{cases}
$$

- RELU is useful for conditional inputs for outputs.

#### Softmax Activation Function

$$
\text{f}(i,j) = \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}}
$$

- Softmax is good for classification due the variability of the outputs of the function.
- Returns a value between [0,1]
- Basically taking the average of exponentiated values.

## NUMPY

- `np.dot(a1, a2)`
  -If matrix, treat this as matrix multiplication. Aka. consider transposition.
  -If vector, treat it as a dot product. Aka. no need to worry about transposition.
- `np.random.randn(row, col)`
  - filled with random floats sampled from a univariate "normal" (Gaussian) distribution of mean 0 and variance 1.
- `np.zeros( (rows, cols) )`
  - populates vector/matrix with 0s
- `np.maximum(int, npArray)`
  - max but does it for every element in the array.
- `np.sum(npArray, axis=None, keepdims=False)`
  - Sums over the specified axis.
- `np.max(inputs, axis=n, keepdims=False)`
  - return the max from the specified axis.
- `np.exp(npArray)`
  - Exponentiates all elements of the npArray.

Common Inputs

- `axis=None` sums over every element in the npArray.
- `axis=0` sums over each column in a _matrix_
- `axis=1` sums over each row in a _matrix_
- `keepdims=True` sum over `axis=n` but keep the same dimensions as input/npArray
