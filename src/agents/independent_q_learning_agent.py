from collections import defaultdict
import random
import numpy as np

from agents.base_agent import BaseAgent

from utils.discretization import discretize


class IndependentQLearningAgent(BaseAgent):

    def __init__(
        self,
        action_space,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
    ):

        self.action_space = action_space

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = {}

    def _initialize_state(self, state):

        if state not in self.q_table:

            self.q_table[state] = [

                0.0

                for _ in range(self.action_space.n)

            ]

    def choose_action(self, observation):

        state = discretize(observation)

        self._initialize_state(state)

        if random.random() < self.epsilon:
            return self.action_space.sample()

        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
    ):
        """
        Updates the Q-table using the Q-Learning equation.
        """

        # Convert observations to discrete states
        state = discretize(observation)
        next_state = discretize(next_observation)

        # Ensure both states exist in the Q-table
        self._initialize_state(state)
        self._initialize_state(next_state)

        # Current estimate
        current_q = self.q_table[state][action]

        # Target value
        if done:

            target = reward

        else:

            max_next_q = max(self.q_table[next_state])

            target = reward + self.gamma * max_next_q

        # Q-Learning update
        self.q_table[state][action] += (
            self.alpha * (target - current_q)
        )