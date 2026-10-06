import gymnasium as gym
import numpy as np
class QLearningAgent:
    def __init__(self, num_bins = 10, state_size = 6, action_size = 3, learning_rate=0.01, discount_factor=0.9,
                 epsilon=1.0, epsilon_decay = 0.995, epsilon_min = 0.1):
        """
        :param state_size: Number of states in the environment
        :param action_size: Number of possible actions
        :param learning_rate: Rate at which the agent will learn
        :param discount_factor: Discount factor
        :param epsilon: Initial exploration rate
        :param epsilon_decay: Rate at which exploration rate will be decayed
        :param epsilon_min: Minimum exploration rate
        """
        self.num_bins = num_bins
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_bounds = [
            (-1, 1), #cosine of theta1
            (-1, 1), #sine of theta1
            (-1, 1), #cosine of theta2
            (-1, 1), #sine of theta2
            (-4 * np.pi, 4 * np.pi), #angular velocity of theta1
            (-9 * np.pi, 9 * np.pi), #angular velocity of theta2
            #theta1 is the angle of the first joint, 0 represents downwards
            #theta2 is the angle of the second joint, relative to the first. 0 represents the same angle
        ]
        self.q_table = np.zeros((self.num_bins, self.action_size))

    def discretize_state(self, state):
        discrete_state = []
        for i, s in enumerate(state):
            lower, upper = self.state_bounds[i]
            #clip s into bounds
            s = np.clip(s, lower, upper)
            #sort s into bins to discretize
            bin_idx = int(s - lower) / ((upper - lower) * (self.num_bins - 1))
            discrete_state.append(bin_idx)
        return tuple(discrete_state)

    def get_state(self, state, training = True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])

env = gym.make('Acrobot-v1', render_mode="human")
observation, info = env.reset()

print(f"Starting observation: {observation}")

for _ in range(5):
    episode_over = False
    total_reward = 0.0

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Total reward: {total_reward}")

    env.reset()

env.close()
