'''
Implement the incomplete functions in the file.

Arguments:
    None
Returns:
    None
'''
import numpy as np
from read_dataset import mnist
import matplotlib.pyplot as plt
import pdb


def sigmoid(scores):
    '''
    calculates the sigmoid of scores
    Inputs: 
        scores array
    Returns:
        sigmoid of scores
    '''

    return 1.0/(1.0+np.exp(-scores))


def step(X, Y, w, b):
    '''
    Implements cost and gradients for the logistic regression with one batch of data
    Inputs:
        X = (n,m) matrix
        Y = (1,m) matrix of labels
        w = (n,1) matrix
        b = scalar
    Returns:
        cost = cost of the batch
        gradients = dictionary of gradients dw and db
    '''
    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)


    # compute the gradients and cost 
    m = X.shape[1]  # number of samples in the batch
    cost1 =  Y*np.log(A)
    cost2 =  (1-Y)*np.log(1-A)

    cost = np.mean(-(cost1+cost2))
    dw = (np.dot(X,((A-Y).T)))/m
    db =  (sum(A-Y))/m
    gradients = {"dw": dw,
                 "db": db}
    return cost, gradients

def optimizer(X, Y, w, b, learning_rate, num_iterations):
    '''
    Implements gradient descent and updates w and b
    Inputs: 
        X = (n,m) matrix
        Y = (1,m) matrix of labels
        w = (n,1) matrix
        b = scalar
        learning_rate = rate at which the gradient is updated
        num_iterations = total number of batches for gradient descent
    Returns:
        parameters = dictionary containing w and b
        gradients = dictionary contains gradeints dw and db
        costs = array of costs over the training 
    '''
    costs = []
    train_error = []
    # update weights by gradient descent
    for ii in range(num_iterations):
        cost, gradients = step(X, Y, w, b)
        dw = gradients["dw"]
        db = gradients["db"]
        w = w - (learning_rate*dw)
        b = b - (learning_rate*b)
        if ii % 10 == 0:
            costs.append(cost)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    parameters = {"w": w, "b": b}
    plt.plot(costs)
    plt.xlabel("iterations")
    plt.ylabel("Error")
    plt.title("Plot of train error vs  iterations")
    plt.show()
    return parameters, gradients, costs

def classify(X, w, b):
    '''
    Outputs the predictions for X

    Inputs: 
        X = (n,m) matrix
        w = (n,1) matrix
        b = scalar

    Returns:
        YPred = (1,m) matrix of predictions
    '''
    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)
    YPred = np.round(A)

    return YPred

    
def main():
    # getting the sbuset dataset from MNIST
    train_data, train_label, test_data, test_label = mnist()
    r,c =train_data.shape

    #normalize the training data
    # min_x = train_data.min(axis =1)
    # max_x = train_data.max(axis =1)
    # min_x = min_x.reshape(-1,1)
    # max_x = max_x.reshape(-1,1)
    # min_x = np.repeat(min_x,c,axis=1)
    # max_x = np.repeat(max_x,c,axis=1)
    # norm_x = (train_data - min_x)/(max_x - min_x)
    # print (train_data - np.mean(train_data, axis=1).reshape(-1,1))
    # print "hi"
    #
    # print np.std(train_data, axis=1).reshape(-1, 1) + 1
    #
    # norm_x = (train_data - np.mean(train_data, axis=1).reshape(-1,1)) / np.std(train_data, axis=1).reshape(-1,1)+1
    # norm_x_test = (test_data - np.mean(test_data, axis=1).reshape(-1, 1)) / np.std(test_data, axis=1).reshape(-1, 1)+1
    # print norm_x
    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = 2000

    # initialize w as array (d,1) and b as a scalar
    w = np.zeros((r,1))
    b = 0

    # learning the weights by using optimize function
    parameters, gradients, costs = optimizer(train_data,train_label, w, b, learning_rate, num_iterations)
    w = parameters["w"]
    b = parameters["b"]
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data,w,b)
    test_Pred = classify(test_data,w,b)
    print train_Pred
    trAcc = np.mean((train_Pred == train_label)) * 100
    teAcc = np.mean((test_Pred == test_label)) * 100
    print("Accuracy for training set is {} %".format(trAcc))
    print("Accuracy for testing set is {} %".format(teAcc))


if __name__ == "__main__":
    main()
