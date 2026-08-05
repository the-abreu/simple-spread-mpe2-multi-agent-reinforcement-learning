from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        return self.action_space.sample()