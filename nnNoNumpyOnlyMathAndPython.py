import numpy as np
import matplotlib.pyplot as plt
import math
import random

# yLength = len(y)
# xLength = len(x)

# w1 = [0.123123122] * xLength
# w2 = [0.75643452423] * xLength
# b1 = [0.151231] * 1
# b2 = [0.3642341] * 1



# W1 = np.array([w1 for x in range(2)])
# W2 = np.array([w2 for x in range(xLength)])
# b1 = np.array([b1 for x in range(2)])
# b2 = np.array([b2 for x in range(xLength)])

# W1 = []
# W2 = []
# b1 = []
# b2 = []
iterations = 10

def transpose(matrix):
    new = []
    for row in zip(*matrix):
        newRow = []
        for item in row:
            newRow.append(item)
        new.append(newRow)
    return new

def sumMatrix(matrix):
    sumVal = 0
    for row in matrix:
        for item in row:
            sumVal += item
    return sumVal

def tanhDerrivMatrix(matrix):
    for rowKey, row in enumerate(matrix):
        for cellKey, item in enumerate(row):
            matrix[rowKey][cellKey] = 1 - (math.tanh(item) ** 2)
    return matrix

def calcLayerForwardPropagation(layer, byMultiple, bias, withTanh):
    z = []
    a = []
    multipleArray = byMultiple
    if type(multipleArray) == int or type(multipleArray) == float :
        multipleArray = [[byMultiple] for e in range(len(layer))]

    for keyRow, w1Row in enumerate(layer):
        z.append([])
        a.append([])

        currentCol = 0
        totalCols = len(multipleArray[0])
        while currentCol < totalCols:
            zNew = 0
            for keyCell,  w1Cell in enumerate(w1Row):

                zNew += w1Cell * multipleArray[keyCell][currentCol]

            if type(bias[keyRow]) == list and len(bias[keyRow]) > 1:
                zNew += bias[keyRow][keyCell]
            else :
                zNew += bias[keyRow][0]
            # print('zNew', zNew)
            aVal = zNew
            if withTanh == True:
                aVal = math.tanh(zNew)

            a[keyRow].append(aVal)
            
            z[keyRow].append(zNew)
            currentCol = currentCol + 1

    return z, a

def calcDotProducts(layer, byMultiple):
    z = []
    multipleArray = byMultiple
    if type(multipleArray) == int or type(multipleArray) == float:
        multipleArray = [[byMultiple]  for e in range(len(layer))]

    for keyRow, w1Row in enumerate(layer):
        z.append([])
        currentCol = 0
        totalCols = len(multipleArray[0])
        while currentCol < totalCols:
            zNew = 0
            for keyCell,  w1Cell in enumerate(w1Row):

                zNew += w1Cell * multipleArray[keyCell][currentCol]
     
            z[keyRow].append(zNew)
            currentCol = currentCol + 1

    return z

def calcMinus(layer, byMinus):
    if (type(layer) == int or type(layer) == float)  and (type(byMinus) == int or type(byMinus) == float):
        return layer - byMinus
    z = []
    for keyRow, w1Row in enumerate(layer):
        z.append([])
        for keyCell,  w1Cell in enumerate(w1Row):
            if type(byMinus) == list:
                if len(byMinus[keyRow]) > 1 :
                    zNew = w1Cell - byMinus[keyRow][keyCell]
                else :
                    zNew = w1Cell - byMinus[keyRow][0]    
            else :
                zNew = w1Cell - byMinus

            z[keyRow].append(zNew)
    return z

def calcMultiple(layer, byMultiple):

    if (type(layer) == int or type(layer) == float)  and (type(byMultiple) == int or type(byMultiple) == float):
        return layer * byMultiple
    z = []

    multipleArray = byMultiple
    if type(multipleArray) == int or type(multipleArray) == float:
        multipleArray = [[byMultiple] for e in range(len(layer))]
    elif  len(byMultiple) < len(layer):
        multipleArray = [byMultiple[0] for e in range(len(layer))]

    if len(layer) < len(multipleArray):
        layerTmp = layer
        byMultiplTmp = byMultiple
        layer = byMultiplTmp
        byMultiple = [layerTmp[0] for e in range(len(byMultiple))]

    for keyRow, w1Row in enumerate(layer):
        z.append([])
        for keyCell,  w1Cell in enumerate(w1Row):
            if type(byMultiple) == list:
                if keyRow in byMultiple :
                    zNew = w1Cell * byMultiple[keyRow][keyCell]
                else :
                    zNew = w1Cell * byMultiple[keyRow][0]
            else :
                zNew = w1Cell * byMultiple

            z[keyRow].append(zNew)
    return z

def forwardPropagation(w1, b1, w2, b2, w3, b3, X):
    z1, a1 = calcLayerForwardPropagation(w1, X, b1, True)

    z2, a2 = calcLayerForwardPropagation(w2, a1, b2, True)

    z3, a3 = calcLayerForwardPropagation(w3, a2, b3, False)

    return z1, a1, z2, a2, z3, a3


def backwardPropagation(w1, w2, w3, b1, b2, b3, z1, a1, z2, a2, z3, a3, Y, Ylist, X):
    # print('Y', Y)
    dz3 = calcMinus(a3, Y)

    dw3 = calcMultiple(dz3, a2)
    db3 = dz3


    dz2 = calcMultiple(calcDotProducts(transpose(w3), dz3), (tanhDerrivMatrix(z2)))
    dw2 = calcDotProducts(dz2, transpose(a1))
    db2 = sumMatrix(dz2)

    dz1 = calcMultiple(calcDotProducts(transpose(w2), dz2), (tanhDerrivMatrix(z1)))
    dw1 = calcDotProducts(dz1, X)
    db1 = sumMatrix(dz1)



    return dw1, dw2, dw3, db1, db2, db3


def initParams() :
    w1 = [[random.uniform(0,1)] for x in range(3)]
    b1 = [[random.uniform(0,1)] for x in range(3)]

    w2 = [[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1) ] for x in range(2)]
    b2 = [[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)] for x in range(2)]


    w3 = [[random.uniform(0,1), random.uniform(0,1)] for x in range(1)]
    b3 = [[random.uniform(0,1)] for x in range(1)]

    # w1 = [
    #     [2],
    #     [4],
    #     [6],
    # ]
    # b1 = [
    #     [1],
    #     [2],
    #     [3],
    # ]

    # w2 = [
    #     [2, 3, 4, ],
    #     [5, 6, 7, ],

    # ]
    # b2 = [
    #     [2],
    #     [3],
    # ]
    

    # w3 = [
    #     [7, 8, ],
    # ]
    # b3 = [
    #     [2],
    # ]

    return w1, w2, b1, b2, w3, b3


def generateX(count, step):
    res = []
    first = -2
    while len(res) <= count:
        res.append(first)
        first = first + step
    
    return res

def predict(w1, b1, w2, b2, w3, b3, X):
    z1, a1, z2, a2, z3, a3 = forwardPropagation(w1, b1, w2, b2, w3, b3, X)
    return a3[0]




X = generateX(20, 0.18)

y = []
for x in X:
    y.append(2*x**3 + 2*x**4 - 4*x**2 + 2)
# y = 2*x**3 + 2*x**4 - 4*x**2 + 2
# x = [x for el in range(5)]
# y = [y for el in range(5)]

# x = np.array(x) 
# y = np.array(y) 
# Y = y / 100
X = X

def learnNetwork(iterations, Xarg, Y):
    plt.ion()
    fig, axs = plt.subplots(1,sharex=True)
    ax = axs

    w1, w2, b1, b2, w3, b3 = initParams()
    learnRate = 0.002

    # lineBlue1, ax, fig = showData(X[0] * 100, Y[0] * 100)
    for i in range(iterations):
        
        for idx, X in enumerate(Xarg):
            z1, a1, z2, a2, z3, a3 = forwardPropagation(w1, b1, w2, b2, w3, b3, X)
            dw1, dw2, dw3, db1, db2, db3 = backwardPropagation(w1, w2, w3, b1, b2, b3, z1, a1, z2, a2, z3, a3, Y[idx], Y, X)

            w1 = calcMinus(w1, calcMultiple(dw1, learnRate))
            w2 = calcMinus(w2, calcMultiple(dw2, learnRate))
            w3 = calcMinus(w3, calcMultiple(dw3, learnRate))
            b1 = calcMinus(b1, calcMultiple(db1, learnRate))
            b2 = calcMinus(b2, calcMultiple(db2, learnRate))
            b3 = calcMinus(b3, calcMultiple(db3, learnRate))


        if i % 100 == 0:
            print(i)
            predictedX = generateX(100, 0.0355)
            predictedY = [ predict(w1, b1, w2, b2, w3, b3,predX)[0] for predX in predictedX]
            ax.clear()
            ax.plot(predictedX, predictedY)
            ax.plot(Xarg , Y)

            plt.pause(0.002)

    return w1, b1, w2, b2


Y = y 
X = X
W1, b1, W2, b2 = learnNetwork(10000, X, Y)