import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data as data
import sys
import ast
import time
import joblib

# df = pd.DataFrame({"x" : x, 'y': y})

class Net():

    z1 = []
    a1 = []
    z2 = []
    a2 = []
    z3 = []
    a3 = []
    z4 = []
    a4 = []
    z5 = []
    a5 = []

    def __init__(self):

        self.w1 = np.random.rand(100,64) / 10
        self.b1 = np.random.rand(100,1) / 10

        self.w2 = np.random.rand(200,100) / 10
        self.b2 = np.random.rand(200,1) / 10
        
        self.w3 = np.random.rand(100,200) / 10
        self.b3 = np.random.rand(100,1) / 10
        
        self.w4 = np.random.rand(64,100) / 10
        self.b4 = np.random.rand(64,1) / 10
        
        self.w5 = np.random.rand(64,64) / 10
        self.b5 = np.random.rand(64,1) / 10
        # b1 = np.ones((3,1))
        # b2 = np.ones((2,1))
        # b3 = np.ones((1,1))


    def forward(self, X):
        z1 = self.w1.dot(X) + self.b1
        a1 = np.tanh(z1)
        
        z2 = self.w2.dot(a1) + self.b2
        a2 = np.tanh(z2)

        z3 = self.w3.dot(a2) + self.b3
        a3 = np.tanh(z3)
        
        z4 = self.w4.dot(a3) + self.b4
        a4 = np.tanh(z4)
        
        z5 = self.w5.dot(a4) + self.b5
        a5 = np.tanh(z5)

        return a5

    def forwardPropagation(self, X):
        self.z1 = self.w1.dot(X) + self.b1
        self.a1 = np.tanh(self.z1)
        
        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = np.tanh(self.z2)

        self.z3 = self.w3.dot(self.a2) + self.b3
        self.a3 = np.tanh(self.z3)
        
        self.z4 = self.w4.dot(self.a3) + self.b4
        self.a4 = np.tanh(self.z4)
        
        self.z5 = self.w5.dot(self.a4) + self.b5
        self.a5 = np.tanh(self.z5)

        # print('z1', z1)

    def backwardPropagation(self, Y, X):
        # print('a2', a2)
        # print('Y', Y)
        dz5 = (self.a5 - Y)

        dw5 = dz5 * self.a4
        db5 = np.sum(dz5)
        # print('dz3', dz3)
        # print('Y', Y)
        # print('dw3', dw3)

        # dz2 = w3.T.dot(dz3) * (1 - (np.tanh(z2) ** 2))
        dz4 = self.w5.T.dot(dz5) * (1 - (np.tanh(self.z4) ** 2))
        dw4 = dz4.dot(self.a3.T)
        db4 = np.sum(dz4)
        
        dz3 = self.w4.T.dot(dz4) * (1 - (np.tanh(self.z3) ** 2))
        dw3 = dz3.dot(self.a2.T)
        db3 = np.sum(dz3)
        
        dz2 = self.w3.T.dot(dz3) * (1 - (np.tanh(self.z2) ** 2))
        dw2 = dz2.dot(self.a1.T)
        db2 = np.sum(dz2)

        dz1 = self.w2.T.dot(dz2) * (1 - (np.tanh(self.z1) ** 2))
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


        return dw1, dw2, dw3, dw4, dw5, db1, db2, db3, db4, db5

    def predict(self, X):
        self.forwardPropagation(X)
        return self.a5

    def learnNetwork(self, Xarg, Y):

        learnRate = 0.01

        for i in range(200):
            
            for idx, X in enumerate(Xarg):
                self.forwardPropagation(Xarg)
                dw1, dw2, dw3, dw4, dw5, db1, db2, db3, db4, db5 = self.backwardPropagation( Y[idx], X)
                dw1 = dw1.reshape(100, 1)
                # print(dw1)
                # print(dw1.shape)
                # print(self.w1.shape)
                self.w1 = self.w1 - (learnRate * dw1)
                self.w2 = self.w2 - (learnRate * dw2)
                self.w3 = self.w3 - (learnRate * dw3)
                self.w4 = self.w4 - (learnRate * dw4)
                self.w5 = self.w5 - (learnRate * dw5)
                self.b1 = self.b1 - (learnRate * db1)
                self.b2 = self.b2 - (learnRate * db2)
                self.b3 = self.b3 - (learnRate * db3)
                self.b4 = self.b4 - (learnRate * db4)
                self.b5 = self.b5 - (learnRate * db5)
                


                if i % 100 == 0:
                    err = (np.sum(Y) - np.sum(self.a5) )
                print('err=' + str(err))
                # print(i)
                # predictedX = np.linspace(-2, 1.5, 100)
                # predictedY = np.array([ predict(w1, b1, w2, b2, w3, b3,predX)[0] for predX in predictedX])
                # ax.clear()
                # ax.plot(predictedX, predictedY)
                # ax.plot(Xarg , Y)

                # plt.pause(0.002)

        # print('a5', self.a5)



        #     break
                
# start_time = time.time()


if sys.argv[1] == 'test':
    x = data.X
    y = data.Y
# else:
#     x = sys.argv[1]
#     x = ast.literal_eval(x)

#     y = sys.argv[2]
#     y = ast.literal_eval(y)
    
# # y = sys.argv[2]
# # print( [5] )
# # print('y arg ', y )

X = np.array(x)
Y = np.array(y)
print('X arg ', len(X) )
print('y arg ', len(Y) )


# net = Net()

# net.learnNetwork(X, Y)

# joblib.dump(net, "./model/chess_model.joblib")
dt = joblib.load("./model/chess_model.joblib")
# dt.learnNetwork(X, Y)

# preds = net.predict(X)
# print("--- %s seconds ---" % (time.time() - start_time))

# # print(np.array2string(preds, separator=','))
# print('preds ', preds )
# print('Y', Y )
print('dt', dt.a5 )

# import chess.engine

# game = chess.Board()

# engine = chess.engine.SimpleEngine.popen_uci("stockfish")
# board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")
# limit = chess.engine.Limit(time=2.0)
# engine.play(board, limit)  

