import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(Z):
    return 1 / 1 + np.exp(-Z)

def sigmoidDerriv(Z):
    return np.exp(-Z) / (1 + np.exp(-Z))**2

x = np.linspace(-2, 1.5, 20)

y = 2*x**3 + 2*x**4 - 4*x**2 + 2
X = np.array(x)
Y = np.array(y)

df = pd.DataFrame({"x" : x, 'y': y})


def initParams():
    # w1 = np.array([
    #     [2],
    #     [4],
    #     [6],
    # ])
    # b1 = np.array([
    #     [1],
    #     [2],
    #     [3],
    # ])

    # w2 = np.array([
    #     [2, 3, 4, ],
    #     [5, 6, 7, ],

    # ])
    # b2 = np.array([
    #     [2],
    #     [3],
    # ])
    

    # w3 = np.array([
    #     [7, 8, ],
    # ])
    # b3 = np.array([
    #     [2],
    # ])



    w1 = np.random.rand(3,1)
    b1 = np.random.rand(3,1)

    w2 = np.random.rand(2,3)
    b2 = np.random.rand(2,1)
    
    w3 = np.random.rand(1,2)
    b3 = np.random.rand(1,1)
    # b1 = np.ones((3,1))
    # b2 = np.ones((2,1))
    # b3 = np.ones((1,1))

    return w1, w2, b1, b2, w3, b3


def forwardPropagation(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = np.tanh(z1)
    
    z2 = w2.dot(a1) + b2
    a2 = np.tanh(z2)

    z3 = w3.dot(a2) + b3
    a3 = z3

    # print('z1', z1)
    # print('a1', a1)
    # print('z2', z2)
    # print('a2', a2)
    # print('z3', z3)
    # print('a3', a3)

    return z1, a1, z2, a2, z3, a3

def backwardPropagation(w1, w2, w3, b1, b2, b3, z1, a1, z2, a2, z3, a3, Y, Ylist, X):
    m = len(Ylist)
    # print('a2', a2)
    # print('Y', Y)
    dz3 = a3 - Y

    dw3 = dz3 * a2
    db3 = dz3
    # print('dz3', dz3)
    # print('Y', Y)
    # print('dw3', dw3)

    # dz2 = w3.T.dot(dz3) * (1 - (np.tanh(z2) ** 2))
    dz2 = w3.T.dot(dz3) * (1 - (np.tanh(z2) ** 2))
    dw2 = dz2.dot(a1.T)
    db2 = np.sum(dz2)

    dz1 = w2.T.dot(dz2) * (1 - (np.tanh(z1) ** 2))
    dw1 = dz1.dot(X)
    db1 = np.sum(dz1)

    # print('dz2', dz2)
    # print('dw2', dw2)
    # print('db2', db2)
    # print('dz1', dz1)
    # print('dw1', dw1)
    # print('db1', db1)
    # dz2 = 
    # print(z1)


    return dw1, dw2, dw3, db1, db2, db3


def predict(w1, b1, w2, b2, w3, b3, X):
    z1, a1, z2, a2, z3, a3 = forwardPropagation(w1, b1, w2, b2, w3, b3, X)
    return a3[0]

def learnNetwork(iterations, Xarg, Y):
    plt.ion()
    fig, axs = plt.subplots(1,sharex=True)
    ax = axs

    w1, w2, b1, b2, w3, b3 = initParams()
    learnRate = 0.002

    ax.plot(df['x'], df['y'])


    # lineBlue1, ax, fig = showData(X[0] * 100, Y[0] * 100)
    for i in range(iterations):
        
        for idx, X in enumerate(Xarg):
            z1, a1, z2, a2, z3, a3 = forwardPropagation(w1, b1, w2, b2, w3, b3, X)
            dw1, dw2, dw3, db1, db2, db3 = backwardPropagation(w1, w2, w3, b1, b2, b3, z1, a1, z2, a2, z3, a3, Y[idx], Y, X)

            w1 = w1 - (learnRate * dw1)
            w2 = w2 - (learnRate * dw2)
            w3 = w3 - (learnRate * dw3)
            b1 = b1 - (learnRate * db1)
            b2 = b2 - (learnRate * db2)
            b3 = b3 - (learnRate * db3)
            


        predictedX = np.linspace(-2, 1.5, 100)
        predictedY = np.array([ predict(w1, b1, w2, b2, w3, b3,predX)[0] for predX in predictedX])
        ax.clear()
        ax.plot(predictedX, predictedY)
        ax.plot(Xarg , Y)

        plt.pause(0.002)

    return w1, b1, w2, b2


        #     break
Y = Y 
X = X
W1, b1, W2, b2 = learnNetwork(1000, X, Y)

