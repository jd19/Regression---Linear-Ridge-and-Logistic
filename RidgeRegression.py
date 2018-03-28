import sys
import numpy as np
from numpy import dot, transpose
from numpy.linalg import inv
import matplotlib.pyplot as plt

def main():
    a = np.load('linRegData.npy')
    X,Y = np.hsplit(a,2)
    actual_x = X
    x = np.ones((100,1))
    for i in range(1,16):
        x = np.append(x,np.power(X,i),axis =1)
    xcopy = x
    ycopy =Y
    x = np.append(x,Y,1)
    np.random.shuffle(x)
    Y = x[:,16:]
    x = np.delete(x,16,1)
    lamda = [0.01,0.05,0.1,0.5,1,5,10]
    min = sys.maxint
    min_lamda = lamda[0]
    all_average_error_test = []
    all_average_error_train = []
    for l in lamda:
        all_error_train = []
        all_error_cv = []
        for i in range(5):
            test_x = x[20*i:20*(i+1),:]
            train_x = np.vstack([x[:20*i,:],x[20*(i+1):,:]])
            test_y = Y[20*i:20*(i+1),:]
            train_y = np.vstack([Y[:20*i,:],Y[20*(i+1):,:]])

            theta = np.dot(np.dot( np.linalg.inv(np.dot(train_x.T,train_x) + ( l ** 2) * np.eye(16) ),train_x.T),train_y)

            error_train = ((np.dot((train_y - np.dot(train_x,theta)).T,(train_y - np.dot(train_x,theta)))) + (l ** 2) * np.dot(theta.T,theta))/80

            error_test = ((np.dot((test_y - np.dot(test_x, theta)).T, (test_y - np.dot(test_x, theta)))) + (l ** 2) * np.dot(theta.T,theta)) / 20
            all_error_train.append(error_train)
            all_error_cv.append(error_test)
        average_error_test = sum(all_error_cv)/5
        average_error_train = sum(all_error_train)/5
        all_average_error_test.append(average_error_test)
        all_average_error_train.append(average_error_train)
        if average_error_test < min:
            min = average_error_test
            min_lamda = l
    print all_average_error_test
    print all_average_error_train
    print min_lamda
    xtranspose = transpose(xcopy)
    product = dot(xtranspose, xcopy)
    termsum = product + np.multiply(np.identity(16), min_lamda**2)
    inverseterm = inv(termsum)
    theta = dot(dot(inverseterm, xtranspose), ycopy)
    pred_final = dot(xcopy, theta)
    error_final = (sum(np.square(pred_final - ycopy)) + min_lamda**2 * np.dot(theta.T,theta))/100
    print error_final
    plt.figure(1)
    plt.scatter(actual_x,ycopy)
    plt.title("plot of polynomial fit to the data")
    plt.plot(actual_x,pred_final)

    plt.figure(2)
    plt.plot(lamda,np.squeeze(all_average_error_test),'r',label ="average_error_test")
    plt.plot(lamda,np.squeeze(all_average_error_train), 'g', label = "average_error_train")
    plt.xlabel("lamda")
    plt.ylabel("error")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()
