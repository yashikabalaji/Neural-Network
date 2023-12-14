import numpy as np
import pandas as pd
import csv
from numpy import genfromtxt
import argparse


def neural_network(x_train, y_train, iterations, learning_rate):
    # Initializing passive weights and biases for the neural network as per the problem
    weight_hidden_a1 = -0.3
    weight_hidden_b1 = 0.4
    weight_hidden_a2 = -0.1
    weight_hidden_b2 = -0.4
    weight_hidden_a3 = 0.2
    weight_hidden_b3 = 0.1
    weight_output_h1 = 0.1
    weight_output_h2 = 0.3
    weight_output_h3 = -0.4
    bias_hidden_1 = 0.2
    bias_hidden_2 = -0.5
    bias_hidden_3 = 0.3
    bias_output = -0.1


    # Function to print placeholders for initial weights and biases
    def print_placeholders(num_placeholders):
        for _ in range(num_placeholders):
            print('-', end=" ")

    # Call this function with the number of placeholders
    print_placeholders(11)

    # Printing the actual values
    print(
        f"{round(bias_hidden_1, 5)} {round(weight_hidden_a1, 5)} {round(weight_hidden_b1, 5)} "
        f"{round(bias_hidden_2, 5)} {round(weight_hidden_a2, 5)} {round(weight_hidden_b2, 5)} "
        f"{round(bias_hidden_3, 5)} {round(weight_hidden_a3, 5)} {round(weight_hidden_b3, 5)} "
        f"{round(bias_output, 5)} {round(weight_output_h1, 5)} {round(weight_output_h2, 5)} {round(weight_output_h3, 5)}"
    )

    for _ in range(iterations):
        for i in range(len(x_train)):
            '''Observable output from sigmoid unit
            # Calculating the net input and output of hidden layer neurons
            # hidden layer input of one node can be calculated using sum of the product of input values and weights with bias
            # input of unit = summation (weights * x) + bias 
            # output function is activation function(sigmoid) of hidden layers'''

            net_hidden_1 = weight_hidden_a1 * x_train[i][0] + weight_hidden_b1 * x_train[i][1] + bias_hidden_1
            out_hidden_1 = 1 / (1 + np.exp(-net_hidden_1))

            net_hidden_2 = weight_hidden_a2 * x_train[i][0] + weight_hidden_b2 * x_train[i][1] + bias_hidden_2
            out_hidden_2 = 1 / (1 + np.exp(-net_hidden_2))

            net_hidden_3 = weight_hidden_a3 * x_train[i][0] + weight_hidden_b3 * x_train[i][1] + bias_hidden_3
            out_hidden_3 = 1 / (1 + np.exp(-net_hidden_3))

            # Net input calculation of output layer and output activation function

            net_output = out_hidden_1 * weight_output_h1 + out_hidden_2 * weight_output_h2 + out_hidden_3 * weight_output_h3 + bias_output
            output = 1 / (1 + np.exp(-net_output))

            # Calculating error to get to know the loss(target output (t) - actual output(o))

            error = (y_train[i] - output)

            '''Chain rule - Backpropagation'''
            # Calculating deltas (changes)for backpropagation
            delta_output = output * (1 - output) * error
            delta_hidden_1 = out_hidden_1 * (1 - out_hidden_1) * (delta_output * weight_output_h1)
            delta_hidden_2 = out_hidden_2 * (1 - out_hidden_2) * (delta_output * weight_output_h2)
            delta_hidden_3 = out_hidden_3 * (1 - out_hidden_3) * (delta_output * weight_output_h3)

            # Weight and bias updates for the output layer using weight of previous iteration, learning rate and output values of hidden node
            weight_output_h1 = weight_output_h1 + learning_rate * delta_output * out_hidden_1
            weight_output_h2 = weight_output_h2 + learning_rate * delta_output * out_hidden_2
            weight_output_h3 = weight_output_h3 + learning_rate * delta_output * out_hidden_3
            bias_output = bias_output + learning_rate * delta_output

            # Weight and bias updates for Hidden Layer 1
            weight_hidden_a1 = weight_hidden_a1 + learning_rate * delta_hidden_1 * x_train[i][0]
            weight_hidden_b1 = weight_hidden_b1 + learning_rate * delta_hidden_1 * x_train[i][1]
            bias_hidden_1 = bias_hidden_1 + learning_rate * delta_hidden_1

            # Weight and bias updates for Hidden Layer 2
            weight_hidden_a2 = weight_hidden_a2 + learning_rate * delta_hidden_2 * x_train[i][0]
            weight_hidden_b2 = weight_hidden_b2 + learning_rate * delta_hidden_2 * x_train[i][1]
            bias_hidden_2 = bias_hidden_2 + learning_rate * delta_hidden_2

            # Weight and bias updates for Hidden Layer 3
            weight_hidden_a3 = weight_hidden_a3 + learning_rate * delta_hidden_3 * x_train[i][0]
            weight_hidden_b3 = weight_hidden_b3 + learning_rate * delta_hidden_3 * x_train[i][1]
            bias_hidden_3 = bias_hidden_3 + learning_rate * delta_hidden_3

            # Printing the current values in the network
            print(
                f"{x_train[i][0]:.5f}\t{x_train[i][1]:.5f}\t{out_hidden_1:.5f}\t{out_hidden_2:.5f}\t{out_hidden_3:.5f}\t{output:.5f}\t{int(y_train[i])}\t"
                f"{delta_hidden_1:.5f}\t{delta_hidden_2:.5f}\t{delta_hidden_3:.5f}\t{delta_output:.5f}\t{bias_hidden_1:.5f}\t{weight_hidden_a1:.5f}\t{weight_hidden_b1:.5f}\t"
                f"{bias_hidden_2:.5f}\t{weight_hidden_a2:.5f}\t{weight_hidden_b2:.5f}\t{bias_hidden_3:.5f}\t{weight_hidden_a3:.5f}\t{weight_hidden_b3:.5f}\t{bias_output:.5f}\t"
                f"{weight_output_h1:.5f}\t{weight_output_h2:.5f}\t{weight_output_h3:.5f}"
            )


if __name__ == "__main__":
    # Parsing arguments passed via command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='CSV file path')

    parser.add_argument('--eta', help='Learning rate')

    parser.add_argument('--iterations', help='Number of iterations')

    args = parser.parse_args()

    file_path = args.data
    learning_rate = float(args.eta)
    iterations = int(args.iterations)

    # Read CSV file and perform data processing and rounding all elements to 5 decimal places
    processed_data = np.genfromtxt(file_path, delimiter=',', autostrip=True)
    processed_data = np.round(processed_data.astype(float), 5)

    # Split into input features (x_train) and target (y_train)
    x_train, y_train = processed_data[:, :-1], processed_data[:, -1]

    neural_network(x_train, y_train, iterations, learning_rate)
