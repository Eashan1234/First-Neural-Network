import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, t_iterations):
        for iteration in range(t_iterations):
            input_layer = training_inputs
            outputs = self.sigmoid(np.dot(input_layer, self.synaptic_weights))

            error = training_outputs - outputs 
            adjustmant = error * self.sigmoid_derivative(outputs)
            self.synaptic_weights += np.dot(input_layer.T, adjustmant)
        
        return outputs

    def think(self, inputs):
        input_layer = inputs.astype(float)
        outputs = self.sigmoid(np.dot(input_layer, self.synaptic_weights))
        return outputs
     

if __name__ == "__main__" :
    neural_network = NeuralNetwork()

    train_in = np.array([[0,0,1],
                         [1,1,1],
                         [1,0,1],
                         [0,1,1]])

    train_out = np.array([[0,1,1,0]]).T

    tr_output = neural_network.train(train_in, train_out, 20000)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    
    input = np.array([A, B, C])
    output = neural_network.think(input)
    print(output)
