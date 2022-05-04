# Single Layer Perceptron (Linear Regression)
# Author: Curren Jeandell

import math
import random
from perceptron import MultiLayerPerceptron


@staticmethod
def generate_training_data(n, function):
    data = []
    for i in range(n):
        x = [[random.uniform(0.0, 3.0)]]
        noise_factor = 0
        y = function(x[0][0], (random.random() * noise_factor, random.random() * noise_factor))
        data.append((x, y))
    return data


def func(example, noise=(0, 0)):
    x = example
    return math.pow(x, 2) #2 * (x + noise[0]) + 0.5 + noise[1]


def main():
    perceptron = MultiLayerPerceptron(2)
    perceptron.add_layer(32, "relu")
    perceptron.add_layer(32, "relu")
    perceptron.add_layer(1, "none", True)
    perceptron.print_edges()
    compare_output(perceptron, func, [[1]])

    training_data = generate_training_data(10000, func)
    test_data = generate_training_data(10000, func)

    training_epochs = 20
    print(f"Mean Squared Error (before training): {perceptron.mean_squared_error(test_data)} | {perceptron.mean_squared_error(training_data)} (Test | Training)")
    perceptron.train(training_data, training_epochs, lambda x: perceptron.step_decay(0.01, 0.9, 5, x), 1, "mean squared error")
    print(f"Mean Squared Error (after {training_epochs} epochs): {perceptron.mean_squared_error(test_data)}")

    compare_output(perceptron, func, [[0]])
    compare_output(perceptron, func, [[0.25]])
    compare_output(perceptron, func, [[0.5]])
    compare_output(perceptron, func, [[2]])
    compare_output(perceptron, func, [[3]])

def compare_output(p, func, input):
    print(f"{input} yielded {p.get_outputs(input)} (expected {func(input[0][0])})")

if __name__ == '__main__':
    main()
