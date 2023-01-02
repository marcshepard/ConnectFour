"""
agent_analysis.py - code that can be tweaked to test the agents

What I've learned:
* Basic agent is *much* stronger if it makes the center-most valid move of moves with equal scores
* (3-level) Minimax agent is much stronger than the basic agent
* RL (reinforced learning) agent is not very strong after 50k training sessions, it beats random
  agent only 85% of the time

To test the rl_agent, you first need to train it
import rl_agent
rl_train(50000, "random") - bootstrap training against the random agent
rl_train(50000, basic_agent) - can incrementally train against one of the non-rl agents
rl_train(50000, rl_train) - TODO - this doesn't seem to work yet (training against itself)
Etc...
After each increment, test it out to see how it's doing

So far results are abysmal (50k iterations against random only beats random 80% of
the time), so I'm putting this on hold
"""

import numpy as np
from kaggle_environments import evaluate, make
from agent import basic_agent, minimax3_agent

def battle(agent1, agent2):
    """battle - pits two agents against for a single game and shows move-by-move GUI
    Useful for debugging new agents (e.g, "battle(agent, 'random')"")"""

    # env = make("connectx", debug=True)
    env.run([agent1, agent2])
    env.render(mode="ipython")

def get_win_percentages(agent1, agent2, n_rounds=100):
    """Pits two agents against for many games and print stats afterwards
    Useful for evaluating new agents against old ones, or tweaks to agents"""
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [],
        n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))


# Test the agents against each other - feel free to modify this as needed to play with things
# Note, one can use built-in "random" to run against an agent that makes random moves
env = make("connectx", debug=True)

print ("Random agent vs basic agent")
get_win_percentages("random", basic_agent, 20)

print ("\nRandom agent vs minimax3")
get_win_percentages("random", minimax3_agent, 20)

print ("\nBasic agent vs minimax3")
get_win_percentages(basic_agent, minimax3_agent, 20)
