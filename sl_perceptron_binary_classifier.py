# Single Layer Perceptron (Binary Classifier)
# Author: Curren Jeandell

import math
import random
from perceptron import Perceptron, Node, InputNode, OutputNode


@staticmethod
def generate_training_data(n, classification_function):
    data = []
    for i in range(n):
        x = [[random.random()], [random.random()]]
        y = classification_function(x)
        data.append((x, y))
    return data

@staticmethod
def classification_func(x):
    return x[1][0] > 0.15 * x[0][0] + 0.3


def main():
    perceptron = Perceptron()

    input_0 = InputNode(perceptron, [1])  # bias input
    input_1 = InputNode(perceptron, [0])
    input_2 = InputNode(perceptron, [0])

    output_0 = OutputNode(perceptron)
    output_0.add_input(input_0)
    output_0.add_input(input_1)
    output_0.add_input(input_2)

    training_data = generate_training_data(10000, classification_func)
    test_data = generate_training_data(10000, classification_func)

    print(f"Accuracy (before training): {perceptron.accuracy(test_data)}")
    perceptron.train(training_data, 20, 0.1, 5)
    print(f"Accuracy (after {20} epochs): {perceptron.accuracy(test_data)}")


if __name__ == '__main__':
    main()
