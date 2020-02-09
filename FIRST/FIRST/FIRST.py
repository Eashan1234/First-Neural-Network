import numpy as np

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def sigmoidDeriv(s):
    return s * (1 - s)

training_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random started synaptic weights: ')
print(synaptic_weights)

for iternation in range(20000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustmants = error * sigmoidDeriv(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustmants)

print ('target: ')
print(training_outputs)

print ('outputs: ')
print(outputs)

print ('error: ')
print(error)

print ('adjustmants: ')
print(adjustmants)

print('New Synaptic Weights: ')
print(synaptic_weights)

print('outputs after training: ')
print(outputs)
