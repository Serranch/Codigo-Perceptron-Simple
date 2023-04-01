import numpy as np

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs + 1)
        
    def predict(self, inputs):
        dot_product = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if dot_product > 0:
            return 1
        else:
            return 0
        
    def train(self, inputs, targets, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = targets[i] - prediction
                self.weights[0] += learning_rate * error
                self.weights[1:] += learning_rate * error * inputs[i]

