import numpy as np
from neuralNetwork import Net
from chessAI import ChessGame
import joblib

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
        dtNet = None
        try:
            dtNet = joblib.load("./model/chess_model.joblib")
            self.nn = dtNet
        except: 
            print('not joblib model found')
        self.nn = Net()
        # print(dtNet)
        # dt.learnNetwork(X, Y)
        print(self.nn.a5[:5])
        # joblib.dump(self.nn, "./model/chess_model.joblib")


    def play_me(self, state):

        return self.nn.forward(state)

def play_game(agent):
    chess = ChessGame()

    play_me = True
    while True:
        action = agent.select_move(play_me.createX())
        # state[action] = current_player
        if play_me:
            me_result = chess.play_me(action)
            if not me_result:
                return 0
        else:
            enemy_result = chess.play_enemy()
            if not enemy_result:
                return 1
        
        play_me = not play_me
            

# Funkcja sprawdzająca, czy nastąpiła wygrana
# def check_win(state, player):
#     win_combinations = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
#     for comb in win_combinations:
#         if all(state[i] == player for i in comb):
#             return True
#     return False

def train_agent(agent, num_episodes, learning_rate=0.01, discount_factor=0.99):

    for episode in range(num_episodes):
        chessGame = ChessGame()
        states_sequence = []
        actions_sequence = []
        rewards_sequence = []

        while True:
            # states_sequence.append(state.copy())

            # action = agent.select_move(state)
            # actions_sequence.append(action)

            # next_state = state.copy()
            # next_state[action] = 1
            reward = 0

            game_result = chessGame.auto_play_chess()


            if game_result:
                reward = 1
            else:
                reward = 0

            rewards_sequence.append(reward)

            if game_result:
                # prev_state = states_sequence[-2]
                # prev_action = actions_sequence[-2]
                # q_values = agent.nn.forward(prev_state)
                # max_next_q = np.max(agent.nn.forward(next_state))
                # q_values[prev_action] += learning_rate * (reward + discount_factor * max_next_q - q_values[prev_action])
                X = np.array(chessGame.history_boards) / 10
                Y = np.array(chessGame.selected_moves) / 10
                agent.nn.learnNetwork(X, Y)

            # state = next_state

            if reward != 0:
                break
        joblib.dump(agent.nn, "./model/chess_model.joblib")

# Przykładowe użycie
if __name__ == "__main__":
    agent = Agent()
    num_episodes = 100
    train_agent(agent, num_episodes)

    # # Testowanie agenta
    # results = {1: 0, 0: 0}  # Wyniki: 1 - wygrana, -1 - przegrana, 0 - remis
    # num_tests = 100
    # for _ in range(num_tests):
    #     result = play_game(agent)
    #     results[result] += 1

    # print(f"Wyniki po {num_tests} grach:")
    # print(f"Wygrane: