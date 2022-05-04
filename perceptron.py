import math
import random
from copy import copy


@staticmethod
def sigmoid(s):
    return 1.0 / (1 + math.pow(math.e, -s))


@staticmethod
def relu(s):
    return max(0, s)


@staticmethod
def d_relu(x):
    if x > 0:
        return 1
    else:
        return 0

@staticmethod
def sgd_warm_restarts(a_min, a_max, current, stepsize):
    return a_min + 0.5 * (a_max - a_min) * (1 + math.cos(current / stepsize * math.pi))


class Perceptron:
    def __init__(self):
        self.edge_weights = {}
        self.layers = []
        self.input_layer = None
        self.output_layer = None

    def add_layer(self, num_nodes, transfer_function_type="sigmoid", output_layer=False):
        """
        Adds a layer to the neural network. This is used for all layers: input, hidden, and output.

        :param num_nodes:
        :param transfer_function_type: sigmoid, relu, or none
        :param output_layer:
        """
        self.layers.append(
            [Node(self, transfer_function_type, [0], f"{chr(ord('a') + len(self.layers))}{i}") for i in
             range(num_nodes)])
        this_layer = self.layers[len(self.layers) - 1]

        previous_layer_to_connect_to = []
        if len(self.layers) == 1:  # input layer
            self.input_layer = this_layer
        else:
            previous_layer_to_connect_to = self.layers[len(self.layers) - 2]
            self.output_layer = this_layer

        for i in range(len(this_layer)):
            node = this_layer[i]
            if i == 0 and not output_layer:  # each layer has one bias node, except output layer
                node.set_input_value([1])
            else:  # bias node doesn't connect to previous layer
                for preceding_node in previous_layer_to_connect_to:
                    node.add_input(preceding_node)

        if output_layer:
            self.initialize_weights()

    def train(self, training_data, epochs, a=lambda x: 0.1, epochs_per_print=0, print_info_function=""):
        """
        Train the network on a list of training examples for a number of epochs. Uses stochastic gradient descent.

        :param training_data:
        :param epochs:
        :param a: learning rate
        :param epochs_per_print:
        :param print_info_function: what info to print. can accept "mean squared error" or "accuracy"
        :return:
        """
        for i in range(0, epochs + 1):
            alpha = a(i)
            for example in training_data:
                x = example[0]
                t = example[1]

                g = self.get_outputs(x)[0]

                self.update_weights(x, alpha, t)
            if epochs_per_print != 0 and i % epochs_per_print == 0:
                if print_info_function == "accuracy":
                    print(
                        f"Epoch {i} Accuracy: {self.classification_accuracy(training_data)} (Training Data) [a={alpha}]")
                elif print_info_function == "mean squared error":
                    print(f"Epoch {i} MQE: {self.mean_squared_error(training_data)} (Training Data) [a={alpha}]")

        print("Training finished.")
        print(''.join(['~' for i in range(10)]))
        self.print_weights()

    def get_outputs(self, input_vectors):
        """
        Feed-forward a set of inputs through the network.
        :param input_vectors:
        :return:
        """
        for i in range(len(input_vectors)):
            input_node = self.input_layer[i + 1]
            input_node.set_input_value(input_vectors[i])

        for n in range(0, len(self.layers)):
            layer = self.layers[n]
            for node in layer:
                node.update_output_value()

        return self.output_layer[0].get_thresholded_output()

    # Stochastic gradient descent
    def update_weights(self, alpha, t):
        """
        Update the network weights through backpropagation. Assumes the training example inputs have already been assigned to the input nodes.
        :param alpha: learning rate
        :param t: actual value of example
        :return:
        """
        for n in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[n]
            for node in layer:
                g = node.get_thresholded_output()[0]

                if node.transfer_function_type == "sigmoid":
                    delta = (g) * (1 - g)
                elif node.transfer_function_type == "relu":
                    delta = d_relu(g)
                else:
                    delta = 1

                if n == len(self.layers) - 1:  # hidden to output weights
                    E = (t - g)
                    delta_j = delta * E
                else:  # input to hidden weights
                    delta_j = 0
                    for outgoing in node.outputs:
                        edge = (node, outgoing)
                        delta_j += delta * self.edge_weights[edge][j] * outgoing.get_delta()
                node.set_delta(delta_j)

                for incoming in node.inputs:
                    edge = (incoming, node)

                    h = incoming.get_thresholded_output()

                    for j in range(len(h)):
                        delta_w = alpha * node.get_delta() * h[j]
                        self.edge_weights[edge][j] += delta_w

    def classification_accuracy(self, test_data):
        correct = 0
        for i in range(len(test_data)):
            example = test_data[i]
            x = example[0]
            t = example[1]

            g = self.get_outputs(x)[0]

            if round(g) == example[1]:
                correct += 1
        return round(correct / float(len(test_data)), 5)

    def mean_squared_error(self, test_data):
        sum_of_squared_errors = 0
        for i in range(len(test_data)):
            example = test_data[i]
            x = example[0]
            t = example[1]

            for j in range(len(x)):
                # skip setting input value of bias node. hence the j+1
                self.input_layer[j + 1].set_input_value(x[j])

            g = self.get_outputs(x)[0]

            sum_of_squared_errors += math.pow(t - g, 2)
        return sum_of_squared_errors / float(len(test_data))

    def step_decay(self, a, drop, stepsize, epoch):
        lrate = a * math.pow(drop, math.floor((1 + epoch) / stepsize))
        return lrate

    def initialize_weights(self):
        for n in range(0, len(self.layers) - 1):
            f_in = len(self.layers[n])
            f_out = len(self.layers[n + 1])

            limit = math.sqrt(6 / float(f_in + f_out))
            print(f_in, f_out)

            for node in self.layers[n]:
                for outgoing in node.outputs:
                    edge = (node, outgoing)
                    self.edge_weights[edge] = [random.uniform(-limit, limit)]

    def print_weights(self):
        print("Trained weights: ")
        for l in range(len(self.layers) - 1):
            layer = self.layers[l]
            for i in range(len(layer)):
                for outgoing in layer[i].outputs:
                    text = f"{layer[i]}->{outgoing}: {self.edge_weights[(layer[i], outgoing)]}"
                    if i == 0:
                        text += " (bias)"
                    print(text)

    def print_edges(self):
        for (n1, n2) in self.edge_weights:
            print(f"{n1} -> {n2}");


class SingleLayerPerceptron(Perceptron):
    def __init__(self, num_input_nodes, num_output_nodes, output_layer_transfer_function_type="sigmoid"):
        Perceptron.__init__(self)
        self.add_layer(num_input_nodes, "none")
        self.add_layer(num_output_nodes, output_layer_transfer_function_type, True)


class MultiLayerPerceptron(Perceptron):
    def __init__(self, num_input_nodes):
        Perceptron.__init__(self)
        self.add_layer(num_input_nodes, "none")


class Node:
    def __init__(self, ann, transfer_function_type="sigmoid", value=[0], name="Node"):
        self.inputs = []
        self.outputs = []
        self.edges = set()
        self.ann = ann
        self.transfer_function_type = transfer_function_type

        self.input_value = value
        self.thresholded_output = 0
        self.delta = 0

        self.name = name

    def __str__(self):
        return self.name

    def add_input(self, input_node):
        edge = (input_node, self)
        if edge not in self.edges:
            self.edges.add(edge)

            if edge not in self.ann.edge_weights:
                self.ann.edge_weights[edge] = 0

            self.inputs.append(input_node)
            input_node.add_output(self)

    def add_output(self, output_node):
        edge = (self, output_node)
        if edge not in self.edges:
            self.edges.add(edge)

            if edge not in self.ann.edge_weights:
                self.ann.edge_weights[edge] = 0

            self.outputs.append(output_node)
            output_node.add_input(self)

    # Maybe should be called get_value()...
    def activate(self):
        self.compute_s()
        if self.transfer_function_type == "sigmoid":
            out = list(map(sigmoid, self.input_value))
        elif self.transfer_function_type == "relu":
            out = list(map(relu, self.input_value))
        else:
            out = self.input_value
        return out

    def compute_s(self):
        if len(self.inputs) == 0:  # tis a bias node
            self.input_value = self.get_input_value()
        else:
            s = 0
            for node in self.inputs:
                s += sum([i * j for (i, j) in zip(node.get_thresholded_output(), self.ann.edge_weights[(node, self)])])
            self.input_value = [s]

    def set_input_value(self, value):
        self.input_value = value

    def get_input_value(self):
        return self.input_value

    def get_thresholded_output(self):
        return self.thresholded_output

    def update_output_value(self):
        self.thresholded_output = self.activate()

    def set_delta(self, delta):
        self.delta = delta

    def get_delta(self):
        return self.delta

    def set_name(self, name):
        self.name = name
