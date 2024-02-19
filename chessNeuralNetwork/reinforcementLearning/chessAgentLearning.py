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
        self.nn = Net()
        self.epsilon = 0.1  # Parametr epsilon-greedy

    def select_action(self, state):
        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            empty_cells = np.where(state == 0)[0]
            return np.random.choice(empty_cells)
        else:
            q_values = self.nn.forward(state)
            return np.argmax(q_values)

# Funkcja grająca w kółko i krzyżyk
def play_game(agent):
    chess = ChessGame()

    while True:
        action = agent.select_action(state)
        state[action] = current_player

        if check_win(state, current_player):
            return current_player

        if np.all(state != 0):
            return 0  # Remis

        current_player = -current_player

# Funkcja sprawdzająca, czy nastąpiła wygrana
def check_win(state, player):
    win_combinations = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for comb in win_combinations:
        if all(state[i] == player for i in comb):
            return True
    return False

# Funkcja trenująca agenta za pomocą metody Q-learning
def train_agent(agent, num_episodes, learning_rate=0.1, discount_factor=0.99):
    for episode in range(num_episodes):
        chessGame = ChessGame()
        states_sequence = []
        actions_sequence = []
        rewards_sequence = []

        while True:
            states_sequence.append(state.copy())

            action = agent.select_action(state)
            actions_sequence.append(action)

            next_state = state.copy()
            next_state[action] = 1
            reward = 0

            if check_win(next_state, 1):
                reward = 1
            elif np.all(next_state != 0):
                reward = 0

            rewards_sequence.append(reward)

            # Aktualizacja wartości Q
            if len(states_sequence) > 1:
                prev_state = states_sequence[-2]
                prev_action = actions_sequence[-2]
                q_values = agent.nn.forward(prev_state)
                max_next_q = np.max(agent.nn.forward(next_state))
                q_values[prev_action] += learning_rate * (reward + discount_factor * max_next_q - q_values[prev_action])
                agent.nn.train(prev_state, q_values)

            state = next_state

            if reward != 0:
                break
        joblib.dump(agent.nn.best_estimator_, "./model/chess_model.joblib")

# Przykładowe użycie
if __name__ == "__main__":
    agent = Agent()
    num_episodes = 10000
    train_agent(agent, num_episodes)

    # Testowanie agenta
    results = {1: 0, -1: 0, 0: 0}  # Wyniki: 1 - wygrana, -1 - przegrana, 0 - remis
    num_tests = 1000
    for _ in range(num_tests):
        result = play_game(agent)
        results[result] += 1

    print(f"Wyniki po {num_tests} grach:")
    # print(f"Wygrane: