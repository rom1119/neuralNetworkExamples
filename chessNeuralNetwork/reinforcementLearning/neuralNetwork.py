import numpy as np
import joblib
import sys
import tensorflow as tf
from tensorflow import keras

class LogEpochScores(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LogEpochScores, self).__init__()

    def on_train_begin(self, logs=None):
        self.model.epoch_log = []

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 500 == 0:
            print(
                  "Up to epoch {}, the average loss is {:7.2f}. acc {:7.2f}".format(epoch, logs["loss"], logs["accuracy"])
            )

        # self.model.epoch_log.append(logs)
class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        if epoch % 500 == 0:
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))



LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (300, 0.01),
    (1500, 0.005),
    (2500, 0.001),
    (4000, 0.0005),
    (5000, 0.0003),
    (6000, 0.0002),
    (7000, 0.0001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


class Net:

    keras_model = None

    def __init__(self, learning_rate=0.01):
        self.input_size = 1
        self.hidden_layer_sizes = [200, 300, 300, 200]
        self.output_size = 3
        self.learning_rate = learning_rate

        self.read_params()

        # input_layer = keras.layers.Input(shape=self.input_size)
        # x = keras.layers.Dense(224, activation="relu")(input_layer)
        # x = keras.layers.Dense(224, activation="relu")(x)
        # x = keras.layers.Dense(256, activation="sigmoid")(x)
        # output_layer = keras.layers.Dense(self.output_size, activation="sigmoid")(x)

        # # losses = {"mse": keras.losses.MeanSquaredError(),
        # #   "mae": keras.losses.MeanAbsoluteError(),
        # #   "huber": keras.losses.Huber()}
        
        # self.keras_model = keras.Model(input_layer, output_layer)
        # self.keras_model.summary()

        opt = tf.keras.optimizers.experimental.SGD(
            learning_rate=0.01,
            name='SGD'
        )

        # self.keras_model.compile(optimizer=opt, loss=tf.losses.MeanSquaredError())
        # initalizer = keras.initializers.VarianceScaling(
        #     mode="fan_avg", distribution="normal"
        # )
        initalizer = 'he_normalV2'
        if self.keras_model is None:
            # Definicja modelu
            self.keras_model = tf.keras.Sequential([
                tf.keras.layers.Dense(200, activation='relu', input_shape=(64,), kernel_initializer=initalizer),
                tf.keras.layers.Dense(400, activation='relu', kernel_initializer=initalizer),
                tf.keras.layers.Dense(400, activation='relu', kernel_initializer=initalizer),
                tf.keras.layers.Dense(200, activation='relu', kernel_initializer=initalizer),
                tf.keras.layers.Dense(66, activation='linear')  # Wyjście o rozmiarze 3
            ])

            ## Kompilacja modelu
            self.keras_model.compile(optimizer=opt,
                        loss='MeanSquaredError',  # lub inna funkcja straty odpowiednia dla Twojego problemu
                        metrics=['accuracy']
                        )

    def read_params(self):
        try:
            # params = joblib.load("./model/chess_model.joblib")
            # self.hidden_weights = params['hidden_weights']
            # self.hidden_biases = params['hidden_biases']
            # self.output_weights = params['output_weights']
            # self.output_bias = params['output_bias']
            # self.keras_model = keras.models.load_model("model/chess_model_not_norm.model")
            self.keras_model = keras.models.load_model("model/chess_modelv2.model")
            pass
        except: 
            print('not  model params found')

    def write_params(self):
        params = {}
        # params['hidden_weights'] = self.hidden_weights
        # params['hidden_biases'] = self.hidden_biases
        # params['output_weights'] = self.output_weights
        # params['output_bias'] = self.output_bias


        # joblib.dump(params, "./model/chess_model.joblib")

        self.keras_model.save("model/chess_modelv2.model")
        # self.keras_model.save("model/chess_model_not_norm.model")

        
    def tanh(self, x):
        return np.tanh(x)
        
    def tanh_derivative(self, x):
        return (1 - (np.tanh(x) ** 2))

    def feedforward(self, X):
        # Propagacja sygnału w przód
        # hidden_output = X
        # self.hidden_outputs = [hidden_output]  # Przechowywanie wyników na warstwach ukrytych
        # for i in range(len(self.hidden_weights)):
        #     hidden_output = self.tanh(np.dot(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
        #     self.hidden_outputs.append(hidden_output)
        # output = np.dot(hidden_output, self.output_weights) + self.output_bias
        # return output
        return self.keras_model.predict(np.array([X]))

    def learnNetwork(self, XArg, yArg):
        loss = 1
        era = 1
        XListLen = len(XArg)
        XList = XArg

        # print(f"X: ")
        # print(XList)
        # print(XList[10])
        # XList = (XList - np.min(XList)) / (np.max(XList) - np.min(XList))
        # print(f"X - NORM: ")
        # print(XList)
        # YList = (YList - np.min(YList)) / (np.max(YList) - np.min(YList))
        # def_learn_rate = self.learning_rate
        # while loss > 0.1:
        YList = yArg
        # print(f"y: ")
        # print(YList[0])
        # print(YList[20])
        # print(YList)
        # return
        self.keras_model.fit(XList, YList, epochs=8000, verbose=0,
                              callbacks=[
                                  LogEpochScores(),
                                    CustomLearningRateScheduler(lr_schedule),

                                         ]
                              )

        # return

        # for i in range(len(XList)):
        #     X = np.array(XList[i])
        #     y = YList[i]
        #     print(f"y: ")
        #     print(y)
        #     print(f"X: ")
        #     print(X)
        #     for epoch in range(num_epochs):
        #         # Propagacja sygnału w przód
        #         output = self.feedforward(X)

        #         # Wsteczna propagacja błędu
        #         self.backpropagation(X, y, output)

        #         if epoch % 400 == 0:
        #             loss = np.mean(np.square(y - output))
        #             print(f"y: ")
        #             print(y)
        #             print(f"output: ")
        #             print(output)
        #             print(f"Epoka: {epoch}, Strata: {loss}")
        #             if loss < 0.0001:
        #                 # print(f"================")

        #                 break
                
        #         # print(f" era: {era}, self.learning_rate: {self.learning_rate} loss: {loss > 0.1}")
        #         # self.learning_rate = self.learning_rate + 0.001
        #         # era = era+1
        #     print(f"output: {i} / {XListLen}")
        #     print(output)
        #     print(f"FINAL ===============  =====Epoka: {epoch}, Strata: {loss}")

        print(f"CHECK ACCURACY ===============  ====")

        for i in range(len(XList)):
            X = XList[i]
            y = YList[i].tolist()
            # print(f"ACCURACY X =")
            # print(f"{X.reshape(8,8)}")

            y_move_idx = YList[i][len(YList[i]) - 1]
            y_def_figure = YList[i][len(YList[i]) - 2]

            del y[len(y) - 1]
            del y[len(y) - 1]
            
            y = np.array(y)
            y = y.reshape(8,8)

            # print(f"y ")
            # print(f"{y} ")
            # print(f"y_move_idx ")
            # print(f"{y_move_idx} ")
            # print(f"y_def_figure ")
            # print(f"{y_def_figure} ")

            # outputY = np.array(self.feedforward(X)) * 10
            predict = np.array(self.feedforward(X)[0]).tolist()

            predict_y_move_idx = np.round(predict[len(predict) - 1], 1) * 10
            predict_y_def_figure = trunc(np.round(predict[len(predict) - 2], 3), 2)

            # print(f"predict bef DEL= {predict}")
            del predict[len(predict) - 1]
            # print(f"predict AF DEL= {predict}")
            del predict[len(predict) - 1]

            predict = np.array(predict)
            outputY = np.round(predict.reshape(8,8),2)

            outputY = trunc(outputY, 1)
            # outputY = np.array(self.feedforward(X)).reshape(8,8)
            

            # print(f"outputY= ")
            # print(outputY)
            # print(f"predict_y_move_idx ")
            # print(f"{predict_y_move_idx} ")
            # print(f"predict_y_def_figure ")
            # print(f"{predict_y_def_figure} ")


        self.write_params()


        # self.learning_rate = def_learn_rate

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
# Przykładowe użycie
# input_size = 64
# hidden_layer_sizes = [300, 300, 300, 300]
# output_size = 3
# if sys.argv[1] == 'test':
#     x = data.X
#     y = data.Y

# X = np.array([x])
# Y = np.array([y])
# print('X arg ', len(X) )
# print('y arg ', len(Y) )

# nn = Net()

# # Przykładowe dane treningowe
# X = np.random.randn(1, input_size)
# y = np.random.randn(1, output_size)

# # Trenowanie sieci
# nn.train(X, y)

