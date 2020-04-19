# myNN.py
# tested with Python3.7

import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MyNode:
    def __init__(self, w):
        self.w = w
    def update(self, dw):
        self.w += dw
    def output(self, x):
         return sigmoid(np.dot(x, self.w))

class MyHiddenLayer:
    def __init__(self, W, b):
        self.b = b
        self.W = W
        self.nbNodes = self.W.shape[0]
    def update(self, dW):
        self.W += dW 
    def output(self, x): 
        y = np.zeros(self.nbNodes) 
        for i in range(0, self.nbNodes): 
            y[i] = sigmoid(np.inner(self.W[i,], x) + self.b)
        return y

class MyOutputLayer: 
    def __init__(self, w): 
        self.w = w 
    def update(self, dw):
        self.w += dw 
    def output(self, x): 
        return np.inner(self.w, x)
        
def example_3nodes_1feature(): 
    import matplotlib.pyplot as plt
    w_hidden = np.array([1.0, -1.5, .5]) 
    w_output = np.array([1, 1, -1])
    hidden_layer = MyHiddenLayer(w_hidden, 5) 
    output_layer =MyOutputLayer(w_output) 
    x = np.linspace(-10, 10, 20)
    y = np.zeros(len(x))
    for i, dx in enumerate(x):
        y[i] = output_layer.output(hidden_layer.output(dx))
        print([y[i]])
    plt.plot(x, y)
    plt.show()

def example_sigmoid():
    import matplotlib.pyplot as plt
    x = np.linspace(-2, 2, 20)
    y = sigmoid(2*x)
    plt.plot(x,y)
    plt.show()

def not_available():
    print("This example does not exist!")
   
if __name__ == "__main__":

    def select_example(x):
        return {
            "a" : "example_3nodes_1feature",
            "b" : "example_sigmoid"
        }.get(x, "not_available")
    
    print("\nSo far, two examples are implemented")
    print("\nExample a: plots the output of a network with one hidden layer, 3 nodes and 1 feature. The purpose is to illustrate what a graph of such a function can look like.")
    print("\nExample b: a plot of a sigmoid function")
    print("(see the script for the code of these examples)")
    example_input = input("\nSelect the example (enter a or b) :")
    example_funct = select_example(example_input)
    locals()[example_funct]()

