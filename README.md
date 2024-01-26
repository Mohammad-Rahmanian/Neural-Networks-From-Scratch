# Computational Intelligence Project: Neural Network Implementation

## About the Project
This project is part of the Computational Intelligence curriculum, and it involves implementing a neural network from scratch without the use of high-level libraries. The main focus is on understanding the inner workings of neural networks, including forward and backward propagation, activation functions, optimization algorithms, and loss computations.

## Project Structure

### Layers
- **Convolution2D**: Implements the 2D convolutional layer for processing two-dimensional data like images.
- **FullyConnected**: A fully connected layer that connects each neuron to all neurons in the previous layer.
- **MaxPooling2D**: A max pooling layer that reduces the spatial size of the input to decrease the amount of parameters and computation.

### Loss Functions
- **BinaryCrossEntropy**: For binary classification tasks, calculates the loss between the predicted and actual labels.
- **MeanSquaredError**: For regression tasks, computes the mean of the squares of the differences between predicted and actual values.

### Optimizers
- **Adam**: An optimizer with an adaptive learning rate that's commonly used for deep neural networks.
- **GradientDescent**: The standard optimization algorithm for training neural networks using gradient descent.

### Activation Functions
- **Sigmoid**: A sigmoid activation function that squashes the input values between 0 and 1.
- **ReLU**: The Rectified Linear Unit (ReLU) activation function that introduces non-linearity into the model.

## Usage
To use this project, you will need to have a basic understanding of Python and neural network concepts. The `housepriceprediction.py` script demonstrates how to apply the neural network for a house price prediction task.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
