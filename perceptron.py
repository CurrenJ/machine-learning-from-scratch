import math
from copy import copy


@staticmethod
def sigmoid(s):
    return 1.0 / (1 + math.pow(math.e, -s))


class Perceptron:
    def __init__(self):
        self.edge_weights = {}
        self.input_nodes = []
        self.output_nodes = []

    def add_input_node(self, input_node):
        self.input_nodes.append(input_node)

    def add_output_node(self, output_node):
        self.output_nodes.append(output_node)

    def train(self, training_data, epochs, a=0.1, epochs_per_print=5):
        for i in range(0, epochs + 1):
            for example in training_data:
                alpha = (epochs - i) / epochs * a
                x = example[0]
                t = example[1]

                for j in range(len(x)):
                    self.input_nodes[j+1].set_value(x[j])

                g = self.output_nodes[0].activate()

                self.update_weights(x, alpha, t, g)
            if i % 5 == 0:
                print(
                    f"Epoch {i} Accuracy: {self.accuracy(training_data)} (Training Data)")

        print("Training finished.")
        print(''.join(['~' for i in range(10)]))
        self.print_weights()

    def update_weights(self, training_example, alpha, t, g):
        for i in range(len(self.input_nodes)):
            input_node = self.input_nodes[i]
            x = [1]
            if i != 0:  # input is always 1 for bias input node
                x = training_example[i - 1]
            for outgoing in input_node.outputs:
                edge = (input_node, outgoing)
                for j in range(len(x)):
                    delta_w = -(t - g) * (g) * (1 - g) * x[j]
                    self.edge_weights[edge][j] += -alpha * delta_w

    def accuracy(self, test_data):
        correct = 0
        for i in range(len(test_data)):
            example = test_data[i]
            x = example[0]
            t = example[1]

            for j in range(len(x)):
                # skip setting input value of bias node. hence the j+1
                self.input_nodes[j + 1].set_value(x[j])

            g = self.output_nodes[0].activate()

            if round(g) == example[1]:
                correct += 1
        return round(correct / float(len(test_data)), 5)

    def print_weights(self):
        print("Trained weights: ")
        for w in range(len(self.input_nodes)):
            for output_node in self.output_nodes:
                text = f"w{w}: {self.edge_weights[(self.input_nodes[w], output_node)]}"
                if w == 0:
                    text += " (bias)"
                print(text)


class Node:
    def __init__(self, ann, default_edge_weight=[0.01]):
        self.inputs = []
        self.outputs = []
        self.edges = set()
        self.ann = ann
        self.default_edge_weight = default_edge_weight

    def add_input(self, input_node):
        edge = (input_node, self)
        if edge not in self.edges:
            self.edges.add(edge)

            if edge not in self.ann.edge_weights:
                self.ann.edge_weights[edge] = copy(self.default_edge_weight)

            self.inputs.append(input_node)
            input_node.add_output(self)

    def add_output(self, output_node):
        edge = (self, output_node)
        if edge not in self.edges:
            self.edges.add(edge)

            if edge not in self.ann.edge_weights:
                self.ann.edge_weights[edge] = copy(self.default_edge_weight)

            self.outputs.append(output_node)
            output_node.add_input(self)

    # Maybe should be called get_value()...
    def activate(self):
        return sigmoid(self.compute_s())

    def compute_s(self):
        s = 0
        for node in self.inputs:
            s += sum([i * j for (i, j) in zip(node.get_value(), self.ann.edge_weights[(node, self)])])
        return s


class InputNode(Node):
    def __init__(self, ann, value=[0]):
        Node.__init__(self, ann)
        self.value = value
        ann.add_input_node(self)

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

class OutputNode(Node):
    def __init__(self, ann):
        Node.__init__(self, ann)
        ann.add_output_node(self)