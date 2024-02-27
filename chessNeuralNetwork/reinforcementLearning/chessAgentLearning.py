import numpy as np
from neuralNetwork import Net
from chessAI import ChessGame
import joblib
import time

# Funkcja aktywacji Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Klasa sieci neuronowej
# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.learning_rate = learning_rate

#         # Inicjalizacja wag
#         self.weights_input_hidden = np.random.randn(input_size, hidden_size)
#         self.weights_hidden_output = np.random.randn(hidden_size, output_size)

#     def forward(self, inputs):
#         # Propagacja w przód
#         hidden_inputs = np.dot(inputs, self.weights_input_hidden)
#         hidden_outputs = sigmoid(hidden_inputs)

#         final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
#         final_outputs = sigmoid(final_inputs)

#         return final_outputs

#     def train(self, inputs, targets):
#         # Propagacja w przód
#         hidden_inputs = np.dot(inputs, self.weights_input_hidden)
#         hidden_outputs = sigmoid(hidden_inputs)

#         final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
#         final_outputs = sigmoid(final_inputs)

#         # Obliczenie błędu
#         output_errors = targets - final_outputs
#         output_delta = output_errors * (final_outputs * (1 - final_outputs))

#         hidden_errors = output_delta.dot(self.weights_hidden_output.T)
#         hidden_delta = hidden_errors * (hidden_outputs * (1 - hidden_outputs))

#         # Aktualizacja wag
#         self.weights_hidden_output += hidden_outputs.T.dot(output_delta) * self.learning_rate
#         self.weights_input_hidden += inputs.T.dot(hidden_delta) * self.learning_rate

# Klasa agenta
class Agent:
    def __init__(self):
        # joblib.dump(net, "./model/chess_model.joblib")
        # dtNet = None
        # try:
        #     dtNet = joblib.load("./model/chess_model.joblib")
        #     self.nn = dtNet
        # except: 
        # print(dtNet)
        self.nn = Net()
        # print('not joblib model found')
        # dt.learnNetwork(X, Y)
        # print(self.nn.a5[:5])
        # joblib.dump(self.nn, "./model/chess_model.joblib")


    def play_me(self, state):

        return np.array(self.nn.feedforward(state))

def play_game(agent, with_ui):
    chess = ChessGame(with_ui)

    play_me = True
    while True:
        X = chess.createX()
        action = agent.play_me(X)[0]
        # state[action] = current_player
        print(f'!!!!X {X}')
        print(f'!!!!action {action}')

        if play_me:
            me_result = chess.play_me(action)
            if not me_result:
                return 0
        else:
            enemy_result = chess.play_enemy()
            if not enemy_result:
                return 1
        # return

        play_me = not play_me
            

def train_agent(agent, num_episodes, learning_rate=0.01, discount_factor=0.99):

    for episode in range(num_episodes):

        rewards_sequence = []

        while True:
            # states_sequence.append(state.copy())

            # action = agent.select_move(state)
            # actions_sequence.append(action)

            # next_state = state.copy()
            # next_state[action] = 1
            reward = 0
            chessGame = ChessGame()

            game_result = chessGame.auto_play_chess()

            if game_result:
                reward = 1
            else:
                reward = 0

            rewards_sequence.append(reward)

            if game_result:
                
                X = np.array(chessGame.history_boards)
                Y = np.array(chessGame.selected_moves)

                # print(X.shape)
                # print(Y.shape)
                # for i in range(len(X)):
                #     print(Y[i])
                #     print(X[i])
                # X = np.array([20, 40, 10, 55, 80])
                # Y = np.array([[40, 80, 20], [20, 30, 20], [50, 90, 10], [10, 10, 10], [40, 80, 20]])
                start_time = time.time()

                agent.nn.learnNetwork(X, Y)

                print("--- %s seconds ---" % (time.time() - start_time))

                # joblib.dump(agent.nn, "./model/chess_model.joblib")

            # state = next_state

            if reward != 0:
                break
        # joblib.dump(agent.nn, "./model/chess_model.joblib")

# Przykładowe użycie
if __name__ == "__main__":
    agent = Agent()
    num_episodes = 1
    # train_agent(agent, num_episodes)

    # # Testowanie agenta
    results = {1: 0, 0: 0}  # Wyniki: 1 - wygrana, -1 - przegrana, 0 - remis
    num_tests = 10
    for _ in range(num_tests):
        result = play_game(agent, True)
        results[result] += 1
    print(f"Wyniki po {num_tests} grach:")
    print(f"Wygrane: {results[1]}")
    print(f"Przegrane: {results[0]}")