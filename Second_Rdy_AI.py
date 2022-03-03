import numpy as np

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
        total = self.weights @ inputs
        return sigmoid(total)

    def backPropogation():
        return 0
        
class NeuralNetwork:
    def __init__(self):
        self.weights = np.array([np.random.random(2) for i in range(3)])
        self.weights2 = np.array([np.random.random(3) for i in range(2)])
        self.weights3 = np.random.random(2)
        self.out_i = [0, 0, 0]
        self.out_h = [0, 0]
        self.out_o = [0]
        bias = []
        self.i = []
        self.h = []
        
        for j in range(3):
            self.i.append(Neuron(self.weights[j], bias))
            
        for j in range(2):
            self.h.append(Neuron(self.weights2[j], bias))
            
        self.o = Neuron(self.weights3, bias)
        
    def updateNeu(self):
        bias = 0
        
        for j in range(3):
            self.i[j] = Neuron(self.weights[j], bias)
            
        for j in range(2):
            self.h[j] = Neuron(self.weights2[j], bias)
            
        self.o = Neuron(self.weights3, bias)
        
    def feedforward(self, x):
        for j in range(3):
            self.out_i[j] = self.i[j].feedforward(x[j])
        
        for j in range(2):
            self.out_h[j] = self.h[j].feedforward(self.out_i)
        
        self.out_o[0] = self.o.feedforward(self.out_h)
        
        return self.out_o[0]

network = NeuralNetwork()
num = 2
x = tr_in[num]
y = tr_out[num]
print(network.feedforward(x), y)

for i in range(10000):
    o = network.feedforward(x)
    bk = o * (1 - o) * (y - o)
    
    for i in range(len(network.weights3)):
        network.weights3[i] += bk * network.out_h[i]
        
    oj = [i for i in network.out_h]
    for i in range(len(network.h)):
        bj = oj[i] * (1 - oj[i]) * sum(sum(network.weights2 * bk))
        for j in range(len(network.weights[i])):
            network.weights2[i][j] += bj * network.out_i[j]
        
    oj = [i for i in network.out_i]
    for i in range(len(network.i)):
        bj = oj[i] * (1 - oj[i]) * sum(sum(network.weights * bk))
        for j in range(len(network.weights[i])):
            network.weights[i][j] += bj * x[i][j]
                
    network.updateNeu()

print(network.feedforward(x), y)