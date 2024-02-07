# Pip Libraries

- `numpy`
- `nnfs`
- `matplotlib`
- `PyQt5`

# Tips

- When you do matrix multiplication, the input and output have the same structure (input[0][0] maps to output[0][0])
- In a weight matrix (assuming transposed), the row represents the number of inputs and the columns represent the number of neurons.

# Vocab

- `Inputs` - The very first 'layer' of the neural networks.
- `Neurons` - The neuron has a weight for each input from the previous layer and has a bias.
- `Layer` - Composed of Neurons or inputs.
- `Layer Outputs` - The number of layer outputs is equal to the number of neurons in that layer.
- `Outputs` - The final outputs of the entire network.
- `pred/prediction` - exact same as outputs, except used in the context of error.
- `Gradient` - a vector that contains the partial derivatives of all inputs of the function.

# Activation Functions

- The activation function is applied after weights AND bias.
- Activation function maps inputs to a value on the function. Hence, the outputs of the actiation function are limited by the definition of the activation function.
- In essense, activation function is the function that is recursively used each `Layer` and can be used to variably map specific
  `Inputs` to specific `Outputs`.

## RELU Activation Function

$$
\text{f}(x) =
\begin{cases}
0 & \text{if } x \leq 0,\\
x & \text{if } x > 0.
\end{cases}
$$

- RELU is useful for conditional inputs for outputs.

## Softmax Activation Function

$$
\text{f}(i,j) = \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}}
$$

- Softmax is good for classification due the variability of the outputs of the function.
- Returns a value between [0,1]
- Basically taking the average of exponentiated values.

# Loss

- Defining error is neccesry to know how well the network is performing.
- Loss is used for optimization.

## Accuracy

- The accuracy metric is used alongside Loss to see how well the network predicts the values.
- Accuracy is calculated by checking whether the index of the largest value in each row of the y_pred matches the index of the largest value in each row of the y_true.

## Categorical Loss

$$
L_i = - \sum_{j} y_{i,j} \log(\hat{y}_{i,j})
$$

- The intution for this error is the following: The outputs of the neural network has a domain of [0,1]. The closer the neural network is to 1, the less loss there is. This makes sense, since $log(1) = 0$.

# Backpropogation

## NUMPY

- `np.array(rawArray)`
  - Turns the rawArray into an _numpy array_ you can use nupy methods on.
  - `npArray.shape` - return the dimensions (e.g (4,3), (1,2,3) ) of the tensor. `len(npArray.shape)` can give you the number of dimensions of the npArray.
  - `npArray1 * npArray2` - multiplys the elements of i1,j1, ... with i2,j1.
  - `npArray.copy()` - deep copy of the npArray.
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
- `np.argmax(npArray, axis=None)`
  - `np.max` but you get the index of the max instead of the maximum value.
- `np.exp(npArray)`
  - Exponentiates all elements of the npArray.
- `np.eye(size)`
  - Returns an _identity matrix_ with dimensions of _size_

Common Inputs

- `axis=None` references every element in the npArray.
- `axis=0` references each column in a _matrix_
- `axis=1` references each row in a _matrix_
- `keepdims=True` operation using `axis=n` but keep the same dimensions as input/npArray
