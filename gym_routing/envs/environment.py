import gym
import numpy as np

from gym_routing.envs import core
from gym_routing.envs.drawer import Drawer

NUMBER_OF_LAYERS = 5


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'none']
    }

    def __init__(self, reward_callback, genData, x_range, y_range, class_list, silent=True):
        self.reward_callback = reward_callback
        self.class_list = class_list
        self.x_range = x_range
        self.y_range = y_range
        self.drawer = Drawer(self.x_range, self.y_range, NUMBER_OF_LAYERS, sum(self.class_list))
        self.world = None
        self.agents = None
        self.n = np.sum(class_list)
        self.action_space = [gym.spaces.Discrete(8)] * self.n
        self.observation_space = gym.spaces.Discrete(1)
        self.genData = genData  # Function to generate initial environment.
        self.silent = silent  # a boolean value
        self.tracks = self.reset()
        self.total_reward = 0

    def step(self, action_n):
        reward = []
        pre_map = np.copy(self.world.map)
        for ind, agent in enumerate(self.agents):
            self._set_action(action_n[ind], agent)
        encounters = self.world.step()
        for ind, agent in enumerate(self.agents):
            reward.append(self.reward_callback(agent, self.world.map[-1], pre_map[-1], encounters[ind]))
        self.total_reward += np.mean(reward)
        self.tracks = [a.state.track for a in self.agents]
        return 0, np.array(reward, dtype=np.uint8), self.world.done(), self.tracks

    def reset(self):
        m, list_of_n_agents, list_of_agents_loc = self.genData(self.x_range, self.y_range, self.class_list)
        self.world = core.World(m, list_of_n_agents, list_of_agents_loc, [self.x_range, self.y_range, NUMBER_OF_LAYERS],
                                self.silent)
        print('World has been made with {} pins, {} nets, and in each nets: {}'.format(sum(list_of_n_agents),
                                                                                       len(list_of_n_agents),
                                                                                       list_of_agents_loc))
        self.net = list_of_n_agents  # shape: [# of class, 1]
        self.agents = self.world.agents
        self.n = len(self.agents)
        self.tracks = np.array([a.state.track for a in self.agents], dtype=np.uint8)
        return self.tracks

    # set env action for a particular agent
    def _set_action(self, action, agent, communication=0):
        action = np.argmax(action)
        agent.action.u = action
        agent.action.c = communication

    # render
    def render(self, mode='none'):
        if mode == 'human':
            self.drawer.set_data(self.tracks, rew=self.total_reward)
            self.drawer.show()
