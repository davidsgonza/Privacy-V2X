"""Q-learning Agent class."""
import logging
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

# Configuración del logger
logger = logging.getLogger(__name__)

class QLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy(), q_table=None, acc_reward=0, enable_logging=True):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = q_table if q_table is not None else {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = acc_reward
        self.enable_logging = enable_logging
        self.print_initial_values()

    def log_info(self, message):
        if self.enable_logging:
            logger.info(message)

    def print_initial_values(self):
        """Print initial values."""
        self.log_info(f"Initial state: {self.state}")
        self.log_info(f"State space: {self.state_space}")
        self.log_info(f"Action space: {self.action_space}")
        self.log_info(f"Alpha: {self.alpha}")
        self.log_info(f"Gamma: {self.gamma}")
        self.log_info(f"Q-table: {self.q_table}")
        self.log_info(f"Exploration strategy: {self.exploration}")
        self.log_info(f"Accumulated reward: {self.acc_reward}")

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        self.log_info(f"Chosen action: {self.action} for state: {self.state}")
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        old_value = self.q_table[s][a]
        new_value = old_value + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - old_value
        )
        self.q_table[s][a] = new_value
        self.state = s1
        self.acc_reward += reward

        self.log_info(f"Learning step - from state: {s} to state: {s1}")
        self.log_info(f"Reward received: {reward}")
        self.log_info(f"Q-value updated from {old_value} to {new_value}")
        self.log_info(f"New state: {self.state}")
        self.log_info(f"Accumulated reward: {self.acc_reward}")

    def get_state(self):
        """Get the current state of the agent."""
        return {
            'q_table': {str(k): v for k, v in self.q_table.items()},
            'acc_reward': self.acc_reward
        }

    def print_q_table(self):
        """Print the Q-table."""
        for state, actions in self.q_table.items():
            self.log_info(f"State {state}: {actions}")
