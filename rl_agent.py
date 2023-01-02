"""
The final agent has no preset rules - rather it uses reinforced learning after battling against
the other agents
Requires pandas, torch, gym, stable_baselines3 (aka sb3) - which don't work on Python 3.11
"""

from sys import version_info
import random
import numpy as np
from os.path import exists
import pandas as pd
import gym

from kaggle_environments import make
from gym import spaces

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def py_ver_ok() -> bool:
    """Check if the python version supports sb3"""
    return version_info[0] == 3 and 8 <= version_info[1] < 11

class ConnectFourGym(gym.Env):  # pylint: disable=too-many-instance-attributes
    """A training environment compatible with sb3, implementing ConnectX as OpenAI gym env"""
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.obs = None
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2, 
                                            shape=(1,self.rows,self.columns), dtype=int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        """Reset"""
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns)

    def change_reward(self, old_reward, done):
        """Change the reward"""
        if old_reward == 1: # The agent won the game
            return 1
        elif done: # The opponent won the game
            return -1
        else: # Reward 1/42
            return 1/(self.rows*self.columns)
    def step(self, action):
        """Check if agent's move is valid"""
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid: # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return np.array(self.obs['board']).reshape(1,self.rows,self.columns), reward, done, _

class CustomCNN(BaseFeaturesExtractor):
    """Neural network for predicting action values"""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """forward"""
        return self.linear(self.cnn(observations))

rl_model = None # pylint: disable=invalid-name
MODEL_DATA_FILE = "rl_model_data_file"

def rl_train(iterations : int, agent = callable):
    """Incrementally train the rl agent's model against a given agent"""
    global rl_model     # pylint: disable=invalid-name, global-statement
    env = ConnectFourGym(agent2=agent)
    if exists (MODEL_DATA_FILE):
        new_rl_model = PPO.load(MODEL_DATA_FILE, env=env)
    else:
        policy_kwargs = dict(features_extractor_class=CustomCNN)
        new_rl_model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

    new_rl_model.learn(total_timesteps=iterations)
    new_rl_model.save(MODEL_DATA_FILE)

    rl_model = new_rl_model

def rl_agent(obs, config):
    """Add interface for the trained agent"""
    global rl_model # pylint: disable=invalid-name, global-statement
    if rl_model is None:
        env = ConnectFourGym()
        rl_model = PPO.load (MODEL_DATA_FILE, env=env)
    # Use the best model to select a column
    col, _ = rl_model.predict(np.array(obs.board).reshape(1, 6,7))
    # If valid move, make it. Else pick a random valid move
    if obs.board[int(col)] == 0:
        return int(col)
    return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
