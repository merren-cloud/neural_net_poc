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

    def query(self, inputs_list) -> numpy.ndarray:

        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.whi, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        

def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print(n.query([1.0, 0.5, -1.5]))
    
if __name__ == "__main__":
    main()
