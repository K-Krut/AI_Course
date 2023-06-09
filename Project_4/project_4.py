import numpy as np

R_MATRIX = np.array([
    [-1, 0, 0, -1, -1, 0, -1, -1],
    [0, -1, 0, 0, -1, -1, -1, -1],
    [0, 0, -1, -1, 0, 0, -1, -1],
    [-1, 0, -1, -1, 0, -1, -1, -1],
    [-1, -1, 0, 0, -1, 0, 0, -1],
    [0, -1, 0, -1, 0, -1, 0, -1],
    [-1, -1, -1, -1, 0, 0, -1, 100],
    [-1, -1, -1, -1, -1, -1, 0, 100],
])

class Agent:
    def __init__(self, r_matrix, end_state, gamma=0.8, epsilon=0.1, attempts=400):
        self.R = r_matrix.copy()
        self.end_state = end_state
        self.gamma = gamma
        self.epsilon = epsilon
        self.attempts = attempts
        self.Q = np.zeros(self.R.shape)
        self.states_l = len(self.R)

    def get_R(self):
        return self.R

    def get_Q(self):
        return self.Q

    def get_end_state(self):
        return self.end_state

    def get_gamma(self):
        return self.gamma

    def get_epsilon(self):
        return self.epsilon

    def get_attempts(self):
        return self.attempts

    def set_attempts(self, atm):
        self.attempts = atm

    def set_gamma(self, g):
        self.gamma = g

    def set_end_state(self, state):
        self.end_state = state

    def set_epsilon(self, eps):
        self.epsilon = eps

    def get_all_moves(self, state):
        return np.where(self.R[state] > -1)[0]

    def random_move(self, state):
        return np.random.choice(self.get_all_moves(state))

    def move(self, state):
        possible_moves = self.get_all_moves(state)
        max_move = np.max(self.Q[state, possible_moves])
        best_moves = [i for i in possible_moves if self.Q[state, i] == max_move]
        return np.random.choice(best_moves)

    def give_reward(self, prev_state, state):
        possible_move = self.get_all_moves(state)
        self.Q[prev_state, state] = self.R[prev_state, state] + self.gamma * np.max(self.Q[state, possible_move])

    def get_next_state(self, state, epsilon):
        return self.random_move(state) if np.random.random() < epsilon else self.move(state)

    def make_attempt(self, state, epsilon=0.0):
        steps = [state]
        while state != self.end_state:
            next_state = self.get_next_state(state, epsilon)
            self.give_reward(state, next_state)
            state = next_state
            steps.append(state)
        return steps

    def train(self, attempts):
        attempts = np.random.randint(0, self.states_l, attempts)
        for j in attempts:
            self.make_attempt(j, self.epsilon)


agent = Agent(R_MATRIX, 7)
agent.train(agent.attempts)
for i in range(agent.states_l):
    print(f"from {i} to {agent.end_state} steps: {', '.join(map(str, agent.make_attempt(i)))}")

