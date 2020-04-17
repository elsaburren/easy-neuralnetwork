#myNNApp
import numpy as np
from myNN import MyHiddenLayer, MyOutputLayer
import matplotlib.pyplot as plt

def main():
    numb_obs = 10
    numb_hidden = 2
    W = np.ones(numb_hidden)
    my_hidden = MyHiddenLayer(W)
    my_output = MyOutputLayer([-1, 1])
   # X = np.array(np.linspace(-10, 10, numb_observations*w.shape[0]).reshape(numb_observations, w.shape[0]))

    x = np.linspace(-10,10,numb_obs)
    x = -1
    print(my_hidden.output(x))
    y = my_output.output(my_hidden.output(x))
    print(y)
    plt.plot(y)
    plt.show()

if __name__ == "__main__":
    main()

