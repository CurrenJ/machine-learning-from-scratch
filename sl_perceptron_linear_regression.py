# Single Layer Perceptron (Linear Regression)
# Author: Curren Jeandell

import math
import random
from perceptron import SingleLayerPerceptron


@staticmethod
def generate_training_data(n, function):
    data = []
    for i in range(n):
        x = [[random.uniform(-10.0, 10.0)]]
        noise_factor = 0
        y = function(x, (random.random() * noise_factor, random.random() * noise_factor))
        data.append((x, y))
    return data


def func(example, noise=(0, 0)):
    x = example[0][0]
    return 2 * (x + noise[0]) + 0.75 + noise[1]


def main():
    perceptron = SingleLayerPerceptron(2, 1, "none")

    training_data = generate_training_data(10000, func)
    test_data = generate_training_data(10000, func)

    training_epochs = 20
    print(f"Mean Squared Error (before training): {perceptron.mean_squared_error(test_data)}")
    perceptron.train(training_data, training_epochs, lambda x: 0.001, 5, "mean squared error")
    print(f"Mean Squared Error (after {training_epochs} epochs): {perceptron.mean_squared_error(test_data)}")

if __name__ == '__main__':
    main()
