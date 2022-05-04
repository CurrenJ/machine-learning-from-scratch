# Single Layer Perceptron (Binary Classifier)
# Author: Curren Jeandell

import math
import random
from perceptron import SingleLayerPerceptron


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
    perceptron = SingleLayerPerceptron(3, 1, "sigmoid")

    training_data = generate_training_data(10000, classification_func)
    test_data = generate_training_data(10000, classification_func)

    print(f"Accuracy (before training): {perceptron.classification_accuracy(test_data)}")
    perceptron.train(training_data, 20, 0.1, 5, "accuracy")
    print(f"Accuracy (after {20} epochs): {perceptron.classification_accuracy(test_data)}")


if __name__ == '__main__':
    main()
