import numpy
import scipy.special
class neuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate) -> None:
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate
        
        self.whi = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self) -> None:
        pass

    def query(self) ->None:
        pass

def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print(n)
if __name__ == "__main__":
    main()
