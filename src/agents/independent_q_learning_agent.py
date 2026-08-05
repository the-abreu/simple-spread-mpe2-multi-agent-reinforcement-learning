from collections import defaultdict
import random

from agents.base_agent import BaseAgent


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

    def choose_action(self, state):

        self._initialize_state(state)

        if random.random() < self.epsilon:

            return self.action_space.sample()

        q_values = self.q_table[state]

        return q_values.index(max(q_values))


    def update(

        self,

        state,

        action,

        reward,

        next_state,

        done,

    ):

        pass