#myNNApp
import numpy as np
from myNN import MyHiddenLayer, MyOutputLayer
import matplotlib.pyplot as plt

def main():
    # write your app here
   
   # example:
    numb_obs = 10
    numb_hidden = 2
    W = np.ones(numb_hidden)
    my_hidden = MyHiddenLayer(W, 1)
    my_output = MyOutputLayer([-1, 1])

    x = np.linspace(-10,10,numb_obs)
    x = -1
    print(my_hidden.output(x))
    y = my_output.output(my_hidden.output(x))
    print(y)
    # end of example
    def sq(x):
        return x*x
    
if __name__ == "__main__":
    main()

