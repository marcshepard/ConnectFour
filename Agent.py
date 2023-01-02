"""
This file creates connect four agent using techniques learned in https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning

There are two agents:
* Basic agent - plays a winning or blocking move if available, else center-most valid move
* Minimax agent - three play look ahead
A third agent will coming, hopefully soon, once torch is ported to Python 3.11):
* ML agent - no upfront rules - learns by playing against other agents

Each agent is a method that takes two parameters, obs and config, that contain the following info:
  obs.mark: the peice assigned to the agent (either 1 or 2)
  obs.board: 42 (=6x7) numpy array representing the connect 4 grid in row-major order, each element is 0 (no play) 1 or 2 (played)
  config.columns: 7 - number of columns in a connect 4 game
  config.rows: 6 - number of rows
  config.inarow: 4 - number in a row needed to win
This interface is used in the kaggle class, and also enables us to use their pre-built kaggle_environments methods for letting the
agents battle it out, so I'm using it here too
"""

import numpy as np
from kaggle_environments import evaluate, make

# The basic agent does a winning move (if it exists), else a blocking move (if it exists), else the center-most valid move
# Note: playing the center-most valid move produces a *substantially better* agent, so I added that to what was in the class
def basic_agent(obs, config):
    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = piece
        return next_grid

    # Returns True if dropping piece in column results in game win
    def check_winning_move(obs, config, col, piece):
        # Convert the board to a 2D grid
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        next_grid = drop_piece(grid, col, piece, config)
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(next_grid[row,col:col+config.inarow])
                if window.count(piece) == config.inarow:
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(next_grid[row:row+config.inarow,col])
                if window.count(piece) == config.inarow:
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(next_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if window.count(piece) == config.inarow:
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(next_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if window.count(piece) == config.inarow:
                    return True
        return False
    
    #########################
    # Agent makes selection #
    #########################
    
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    for col in valid_moves:
        # If we have a winning move, take it
        if check_winning_move(obs, config, col, obs.mark):
            return col
        # Else if they have a winning move, block it
        if check_winning_move(obs, config, col, 3-obs.mark):
            return col

    # Else chose the middle-most valid move
    return valid_moves[len(valid_moves)//2]     # Performs better than random.choice(valid_moves)

# The minimax3 agent does 3 step lookahead; picking the move that looks like it will have the best result after the opponent
# makes their best move and then the agent makes a subsequent best move
def minimax3_agent(obs, config):
    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows
    
    # Helper function for minimax: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
        return score
    
    # Uses minimax to calculate value of dropping piece in selected column
    def score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = minimax(next_grid, nsteps-1, False, mark, config)
        return score

    # Helper function for minimax: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False

    # Minimax implementation
    def minimax(node, depth, maximizingPlayer, mark, config):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return get_heuristic(node, mark, config)
        if maximizingPlayer:
            value = -np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, minimax(child, depth-1, False, mark, config))
            return value
        else:
            value = np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                value = min(value, minimax(child, depth-1, True, mark, config))
            return value
    
    # Algo
    N_STEPS = 3
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select a column that maximizes the min possible score (if there is a tie, take the middle column)
    return max_cols[len(max_cols)//2]   # Performs better than random.choice(max_cols)

# The final agent has no preset rules - rather it uses reinforced learning after battling against the other agents
# Requires pip install pandas, torch, gym, stable_baselines3 - which don't work on Python 3.11
# TODO - this is skeletal code from the class; I've not yet tried and debugged it since I'm on Python 3.11 and haven't down-graded
from sys import version_info
def py_ver_ok():
    return version_info[0] == 3 and 8 <= version_info[1] < 11

model_data = "sb3_model_data"   # It takes forever to train an sb3 model, so need a place to save it when done

if False and py_ver_ok():
    import gym
    from gym import spaces
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO 
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import random

    # To prep for ML training, we first need to create an environment compatible with Stable Baselines.
    # For this, we define the ConnectFourGym class below. This class implements ConnectX as an OpenAI Gym environment
    class ConnectFourGym(gym.Env):
        def __init__(self, agent2="random"):
            ks_env = make("connectx", debug=True)
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
            self.obs = self.env.reset()
            return np.array(self.obs['board']).reshape(1,self.rows,self.columns)
        def change_reward(self, old_reward, done):
            if old_reward == 1: # The agent won the game
                return 1
            elif done: # The opponent won the game
                return -1
            else: # Reward 1/42
                return 1/(self.rows*self.columns)
        def step(self, action):
            # Check if agent's move is valid
            is_valid = (self.obs['board'][int(action)] == 0)
            if is_valid: # Play the move
                self.obs, old_reward, done, _ = self.env.step(int(action))
                reward = self.change_reward(old_reward, done)
            else: # End the game and penalize agent
                reward, done, _ = -10, True, {}
            return np.array(self.obs['board']).reshape(1,self.rows,self.columns), reward, done, _

    # Neural network for predicting action values
    class CustomCNN(BaseFeaturesExtractor):
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
            return self.linear(self.cnn(observations))

    # This method trains the rl_agent and persists the model to the given file_name so we can later load it
    def train_rl_agent():  
        # We'll train out reenforced learning agent by letting it play against the basic agent
        env = ConnectFourGym(agent2=basic_agent)

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
        )
                
        # Initialize agent
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

        # Train agent
        model.learn(total_timesteps=60000)
        model.save(model_data)
        return True

    rl_model = None

    # Add interface for the trained agent
    def rl_agent1(obs, config):
        if rl_model is None:
            env = ConnectFourGym()
            rl_model = PPO.load (model_data, env=env)
        # Use the best model to select a column
        col, _ = rl_model.predict(np.array(obs['board']).reshape(1, 6,7))
        # Check if selected column is valid
        is_valid = (obs['board'][int(col)] == 0)
        # If not valid, select random move. 
        if is_valid:
            return int(col)
        else:
            return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])


