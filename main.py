import gymnasium as gym
import numpy as np
class QLearningAgent:
    def __init__(self, num_bins = 10, state_size = 6, action_size = 3, learning_rate=0.2, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay = 0.9995, epsilon_min = 0.05):
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
            (-4 * np.pi, 4 * np.pi), #angular velocity of theta2
            #theta1 is the angle of the first joint, 0 represents downwards
            #theta2 is the angle of the second joint, relative to the first. 0 represents the same angle
        ]
        self.q_table = np.zeros(((self.num_bins,) * self.state_size + (self.action_size,)))

    def discretize_state(self, state):
        discrete_state = []
        for i, s in enumerate(state):
            lower, upper = self.state_bounds[i]
            #clip s into bounds
            s = np.clip(s, lower, upper)
            #sort s into bins to discretize
            bin_idx = int((s - lower) / (upper - lower) * (self.num_bins - 1))
            discrete_state.append(bin_idx)
        return tuple(discrete_state)

    def get_action(self, state, training = True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])

    def update(self, state, action, reward, next_state, terminal):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        current_q = self.q_table[discrete_state][action]

        if terminal:
            best_next_q = 0
        else:
            best_next_q = max(self.q_table[discrete_next_state])

        self.q_table[discrete_state][action] = (1 - self.learning_rate) * current_q + self.learning_rate * ((self.discount_factor * best_next_q) + reward)
        pass

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(num_episodes = 1000, render = False):
    env = gym.make('Acrobot-v1', render_mode = 'human' if render else None)
    agent = QLearningAgent()
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_over = False
        while not episode_over:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            episode_over = done
            total_reward += reward
            state = next_state

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    print(f"\nTraining complete!\n")
    return agent, episode_rewards

def test_agent(agent, num_episodes = 5, render = False):
    env = gym.make('Acrobot-v1', render_mode = 'human' if render else None)
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_over = False
        while not episode_over:
            action = agent.get_action(state, False)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = truncated or terminated
            state = next_state

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Reward: {total_reward:.2f}")

def main():
    agent, rewards = train_agent(num_episodes=6000)
    test_agent(agent, num_episodes = 5, render = True)

main()