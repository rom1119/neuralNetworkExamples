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
        if epoch % 100 == 0:
            print(
                  "Up to epoch {}, the average loss is {:7.2f}.".format(epoch, logs["loss"])
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
        if epoch % 100 == 0:

            if not hasattr(self.model.optimizer, "lr"):
                raise ValueError('Optimizer must have a "lr" attribute.')
            # Get the current learning rate from model's optimizer.
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            # Call schedule function to get the scheduled learning rate.
            scheduled_lr = self.schedule(epoch, lr)
            # Set the value back to the optimizer before this epoch starts
            tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))



LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (300, 0.01),
    (1000, 0.005),
    (2000, 0.001),
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

        # Inicjalizacja wag i biasów dla warstw ukrytych
        self.hidden_weights = [np.random.randn(self.input_size, self.hidden_layer_sizes[0])]
        self.hidden_biases = [np.zeros((1, self.hidden_layer_sizes[0]))]
        for i in range(1, len(self.hidden_layer_sizes)):
            weight = np.random.randn(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i])
            bias = np.zeros((1, self.hidden_layer_sizes[i]))
            print(f'L={i} weight={weight.shape} bias={bias.shape}')
            self.hidden_weights.append(weight)
            self.hidden_biases.append(bias)

        # Inicjalizacja wag i biasów dla warstwy wyjściowej
        self.output_weights = np.random.randn(self.hidden_layer_sizes[-1], self.output_size)
        self.output_bias = np.zeros((1, self.output_size))

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
            ## Definicja modelu
            self.keras_model = tf.keras.Sequential([
                tf.keras.layers.Dense(200, activation='relu', input_shape=(1,), kernel_initializer=initalizer),
                tf.keras.layers.Dense(400, activation='relu', kernel_initializer=initalizer),
                tf.keras.layers.Dense(400, activation='relu', kernel_initializer=initalizer),
                tf.keras.layers.Dense(200, activation='relu', kernel_initializer=initalizer),
                tf.keras.layers.Dense(3, activation='linear')  # Wyjście o rozmiarze 3
            ])

            ## Kompilacja modelu
            self.keras_model.compile(optimizer=opt,
                        loss='mean_absolute_error',  # lub inna funkcja straty odpowiednia dla Twojego problemu
                        )

    def read_params(self):
        try:
            # params = joblib.load("./model/chess_model.joblib")
            # self.hidden_weights = params['hidden_weights']
            # self.hidden_biases = params['hidden_biases']
            # self.output_weights = params['output_weights']
            # self.output_bias = params['output_bias']
            self.keras_model = keras.models.load_model("model/chess_model_not_norm.model")
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

        self.keras_model.save("model/chess_model_not_norm.model")

        
    def tanh(self, x):
        return np.tanh(x)
        
    def tanh_derivative(self, x):
        return (1 - (np.tanh(x) ** 2))

    def feedforward(self, X):
        # Propagacja sygnału w przód
        hidden_output = X
        self.hidden_outputs = [hidden_output]  # Przechowywanie wyników na warstwach ukrytych
        for i in range(len(self.hidden_weights)):
            hidden_output = self.tanh(np.dot(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
            self.hidden_outputs.append(hidden_output)
        output = np.dot(hidden_output, self.output_weights) + self.output_bias
        return output

    def backpropagation(self, X, y, output):
        # Obliczenie błędu na warstwie wyjściowej
        output_error = y - output
        output_delta = output_error * self.tanh_derivative(output)

        # Propagacja wsteczna błędu
        hidden_errors = []
        hidden_deltas = []
        hidden_errors.insert(0, output_delta.dot(self.output_weights.T))
        hidden_deltas.insert(0, hidden_errors[0] * self.tanh_derivative(self.hidden_outputs[-1]))

        for i in range(len(self.hidden_weights) - 1, 0, -1):
            hidden_errors.insert(0, hidden_deltas[0].dot(self.hidden_weights[i].T))
            hidden_deltas.insert(0, hidden_errors[0] * self.tanh_derivative(self.hidden_outputs[i]))

        # Aktualizacja wag i biasów
        self.output_weights += self.hidden_outputs[-1].T.dot(output_delta) * self.learning_rate
        self.output_bias += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] += self.hidden_outputs[i].T.dot(hidden_deltas[i]) * self.learning_rate
            self.hidden_biases[i] += np.sum(hidden_deltas[i], axis=0, keepdims=True) * self.learning_rate

    def learnNetwork(self, XArg, yArg, num_epochs=1000):
        loss = 1
        era = 1
        XListLen = len(XArg)
        XList = XArg

        print(f"X: ")
        print(XList)
        # XList = (XList - np.min(XList)) / (np.max(XList) - np.min(XList))
        # print(f"X - NORM: ")
        # print(XList)
        # YList = (YList - np.min(YList)) / (np.max(YList) - np.min(YList))
        # def_learn_rate = self.learning_rate
        # while loss > 0.1:
        YList = yArg
        print(f"y: ")
        print(YList)
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
            y = np.array(YList[i])
            print(f"ACCURACY X = {X} y= {y}")

            # outputY = np.array(self.feedforward(X)) * 10
            outputY = np.array(self.keras_model.predict(X.reshape((-1, 1))))
            
            print(f"y = {y} outputY= {outputY}")


        self.write_params()


        # self.learning_rate = def_learn_rate


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

