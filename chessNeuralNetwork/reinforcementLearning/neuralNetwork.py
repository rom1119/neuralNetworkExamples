import numpy as np

class Net:
    def __init__(self, learning_rate=0.08):
        self.input_size = 64
        self.hidden_layer_sizes = [200, 300, 300, 200]
        self.output_size = 3
        self.learning_rate = learning_rate

        # Inicjalizacja wag i biasów dla warstw ukrytych
        self.hidden_weights = [np.random.randn(self.input_size, self.hidden_layer_sizes[0])]
        self.hidden_biases = [np.zeros((1, self.hidden_layer_sizes[0]))]
        for i in range(1, len(self.hidden_layer_sizes)):
            weight = np.random.randn(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i])
            bias = np.zeros((1, self.hidden_layer_sizes[i]))
            self.hidden_weights.append(weight)
            self.hidden_biases.append(bias)

        # Inicjalizacja wag i biasów dla warstwy wyjściowej
        self.output_weights = np.random.randn(self.hidden_layer_sizes[-1], self.output_size)
        self.output_bias = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def tanh(self, x):
        return np.tanh(x)

    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def tanh_derivative(self, x):
        return (1 - (np.tanh(x) ** 2))

    def feedforward(self, X):
        # Propagacja sygnału w przód
        hidden_output = X
        self.hidden_outputs = [hidden_output]  # Przechowywanie wyników na warstwach ukrytych
        for i in range(len(self.hidden_weights)):
            hidden_output = self.tanh(np.dot(hidden_output, self.hidden_weights[i]) + self.hidden_biases[i])
            self.hidden_outputs.append(hidden_output)
        output = self.tanh(np.dot(hidden_output, self.output_weights) + self.output_bias)
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

    def learnNetwork(self, XArg, yArg, num_epochs=10000):
        loss = 1
        era = 1
        XList = XArg / 10
        YList = yArg / 10
        def_learn_rate = self.learning_rate
        # while loss > 0.1:
        for i in range(len(XList)):
            X = XList[i]
            y = YList[i]
            for epoch in range(num_epochs):
                # Propagacja sygnału w przód
                output = self.feedforward(X)

                # Wsteczna propagacja błędu
                self.backpropagation(X, y, output)

                if epoch % 200 == 0:
                    loss = np.mean(np.square(y - output))
                    if loss < 0.001:
                        # print(f"================")
                        # print(f"output: ")
                        # print(output)
                        # print(f"y: ")
                        # print(y)
                        print(f"Epoka: {epoch}, Strata: {loss}")

                        break
                
                # print(f" era: {era}, self.learning_rate: {self.learning_rate} loss: {loss > 0.1}")
                # self.learning_rate = self.learning_rate + 0.001
                # era = era+1
            print(f"FINAL ===============  =====Epoka: {epoch}, Strata: {loss}")

        print(f"CHECK ACCURACY ===============  ====")

        for i in range(len(XList)):
            X = XList[i]
            y = np.array(YList[i]) * 10

            outputY = np.array(self.feedforward(X)) * 10
            
            print(f"y = {y} outputY= {outputY}")




        # self.learning_rate = def_learn_rate


# Przykładowe użycie
# input_size = 64
# hidden_layer_sizes = [300, 300, 300, 300]
# output_size = 3
# nn = Net(input_size, hidden_layer_sizes, output_size)

# # Przykładowe dane treningowe
# X = np.random.randn(1, input_size)
# y = np.random.randn(1, output_size)

# # Trenowanie sieci
# nn.train(X, y)