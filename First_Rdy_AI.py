import numpy as np
from random import randint

tr_in = np.array([[[0,0],[1,1],[0,0]],
                  [[1,1],[1,0],[0,1]],
                  [[1,1],[1,0],[1,0]],
                  [[0,0],[0,1],[1,0]],
                  [[0,0],[1,1],[1,1]],
                  [[0,0],[1,0],[1,1]]])

tr_out = np.array([0,1,1,0,0,0])

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def feedforward(self, inputs):
        total = self.weights @ inputs + self.bias
        return sigmoid(total)

    def backPropogation():
        return 0
        
    
class NeuralNetwork:
    def __init__(self):
        self.weights = np.array([np.random.random(2) for i in range(3)])#np.array([1, 0, 0])
        self.weights2 = np.random.random(3)
        self.out_i = [0, 0, 0]
        self.out_o = [0]
        bias = 0.5
        self.i = []
        
        for j in range(3):
            self.i.append(Neuron(self.weights[j], bias))
        
        self.o = Neuron(self.weights2, bias)
        
    def updateNeu(self):
        bias = 0
        for j in range(3):
            self.i[j] = Neuron(self.weights[j], bias)
        
        self.o = Neuron(self.weights2, bias)
    
    def feedforward(self, x):
        for j in range(3):
            self.out_i[j] = self.i[j].feedforward(x[j])
                
        #print(out_o1, out_o2)
        
        self.out_o[0] = self.o.feedforward(np.array(self.out_i))
        
        return self.out_o[0]



network = NeuralNetwork()
num = 5
x = tr_in[num]
y = tr_out[num]
print(network.feedforward(x), y)

for i in range(10000):
    o = network.feedforward(x)
    bk = o * (1 - o) * (y - o)
    
    for i in range(len(network.weights2)):
        network.weights2[i] += bk * network.out_i[i]
                
    oj = [i for i in network.out_i]
    for i in range(len(network.i)):
        bj = oj[i] * (1 - oj[i]) * sum(sum(network.weights * bk))
        for j in range(len(network.weights[i])):
            network.weights[i][j] += bj * x[i][j]
                
    network.updateNeu()

print(network.feedforward(x), y)