import numpy as np
from numpy import dot, transpose
from numpy.linalg import inv
import matplotlib.pyplot as plt

def main():
    a = np.load('linRegData.npy')
    X,Y = np.hsplit(a,2)
    x = np.append(np.ones((100,1)),X,axis =1)
    xtranspose = transpose(x)
    product = dot(xtranspose,x)
    inverseterm = inv(product)
    theta = dot(dot(inverseterm,xtranspose),Y)
    pred = dot(x,theta)
    plt.scatter(X,Y)
    plt.plot(X,pred)
    plt.title("Plot of line fit to the data")
    error = sum(np.square(pred-Y))/100
    print "error :",error
    plt.show()


if __name__ == "__main__":
    main()
