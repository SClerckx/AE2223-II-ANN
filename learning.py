import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Union

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    
class BigOofNetwork(object):
    """ Class that will contain implementation of ANN. """
    def __init__(self, lsizes: List[int]) -> None: 
        # Determine number of layers and save layer sizes
        self.num_layers: int = len(lsizes)
        self.lsizes = lsizes
        # Generate random biases and weights for all layers except input layer
        # We use the np.random.randn() function to generate a matrix of size (y, 1) with random values
        #   where y is size of respective layer.
        self.biases = [np.random.randn(y, 1) for y in lsizes[1:]]
        # We use the same function again, however this time we make a matrix of size (y, x)
        # where x and y are the sizes of consecutive layers starting from the second layer.
        # e.g. x=len(2nd layer),y=(3rd layer) and so forth.
        self.weights = [np.random.randn(y, x) for x, y in zip(lsizes[:-1], lsizes[1:])]
        
    def train_network(self, training_data, 
        epochs: int, mini_batch_size: int, learning_rate: float,
        test_data=None) -> None:
        """ Main function to train the network using backpropagation. """
        # Initialise list to keep track of the accuracy of the neural network at identifying
        # positive matches.
        self.accuracies = []
        # First we ensure that the input data is turned into a Python list and save the length.
        training_data = list(training_data)
        n = len(training_data)
        # If there is test data, do the same as above
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        # Run loop for every epoch
        for c in range(epochs):
            # Shuffle the data to introduce randomness
            random.shuffle(training_data)
            # Divide data into mini-batches
            mini_batches = [training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # For every batch, run the update_mini_batch helper function
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                # Append evaluated accuracy and print progress to console
                accuracy = self.evaluate(test_data)
                self.accuracies.append(accuracy)
                print("Epoch {} : {} / {}".format(c,accuracy,n_test))
            else:
                print("Epoch {} complete".format(c))
    
    def update_mini_batch(self, mini_batch, learning_rate: int) -> None:
        # Initialise empty gradient matrices for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # For each piece of data in the batch,...
        for x, y in mini_batch:
            # Run backpropagation to get change in gradient,
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # and apply gradient and save results in original matrix.
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Update weights and biases using gradients calculated
        self.weights = [w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ A fundamental function which performs backpropagation and returns
        the gradient for the cost function."""
        # Initialise derivative of cost to bias and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Implement feed forward
        activation = x
        # List to store all the activations, layer by layer
        activations = [x]
        # List to store all the z vectors, layer by layer
        zs = []
        # Run loop for bias and weight matrix in each layer
        for b, w in zip(self.biases, self.weights):
            # Get dot product of weights and activation of previous layer
            # and add bias to it. Append the value of z to zs and calculate new 
            # activation for the nodes (using sigmoid) and append this as well.
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Implement backward pass
        # Calculate cost derivate starting from the last layer and find delta
        # by multiplying derivative by sigmoid prime
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # Save the values in their respective lists
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Continue the same procedure for the remaining layers, going backwards.
        # Negative indices are used here to address indices starting from the last one.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        # Return the final list of cost derivatives as required
        return (nabla_b, nabla_w)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        # Evaluates result using neural network biases and weights.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def evaluate(self, test_data) -> int:
        """ Function that returns the number of inputs for which the 
        network gave the expected result"""
        # For each piece of data, we get the largest value in the network's
        # feedforward results and check to see if they match the expected result.
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of derivatives for the output activations."""
        return (output_activations-y)

    def plot_accuracies(self) -> None:
        """ Plots the accuracies achieved during the last training session"""
        plt.plot(np.arange(1., len(self.accuracies)+1, 1.), self.accuracies)
        plt.xlabel("Epochs"), plt.ylabel("Accuracy [%]"), plt.show()