"""
AgentAnalysis.py - code that can be tweaked to test the agents

What I've learned in the process:
* The basic and minimax agents perform better if they make the center-most valid move among among moves with equal scores
"""

from Agent import evaluate, make, basic_agent, minimax3_agent, np

# battle - pits two agents against for a single game and shows move-by-move GUI
# Useful for debugging new agents (e.g, "battle(agent, 'random')"")
def battle(agent1, agent2):
    env.run([agent1, agent2])
    env.render(mode="ipython")

# get_win_percentage - pits two agents against for many games and print stats afterwards
# Useful for evaluating new agents against old ones, or tweaks to agents
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))


# Test the agents against each other - feel free to modify this as needed to play with things
# Note, one can use built-in "random" to run against an agent that makes random moves
env = make("connectx", debug=True)
get_win_percentages("random", minimax3_agent, 20)
