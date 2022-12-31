This app plays connect four.

You can either install it yourself (see install instructions below), or play it from https://replit.com/@marcshepard1/ConnectFour?embed=true 

To play: click/tap to select a column, double-click/double-tap to drop at that column

It consists of the following files:
* ConnectFour.pyw - A graphical use interface for interactive play and the main launch point for the app
* Agent.py - Agents the user can play against. These agents were created using the techniques learned in https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning
* AgentAnalysis.py - Code for evaluating the agents, as well as summary comments on what was learned

Installation and running this program requires Python; I've only tested it with 3.8.12 and 3.11 - so any version in that range should work
You also need to install the following packages:
* numpy
* kaggle_environments

Note: this app will eventually also require these packages for an ML agent (which uses stable-baselines3 for reinforced learning):
* pandas
* gym
* torch
* stable-baselines3
However I've not yet added that agent because I'm using Python 3.11 and torch doesn't yet run on 3.11 (nor does tensorflow, BTW).
These should be coming soon, so I'll wait rather than downgrade my python kaggle_environments

Note on agents:
* Basic agent - wins close to 100% of the time against random agent
* Minimax_agent - wins about 75% of the time against basic agent (half wins, half ties), and 100% against random agent
* rl_agent trained against basic agent w 50k training cycles - wins 70% of the time against random agent (and loses 30%), and less than 50% against basic_agent, 0% vs minimax agent
* rl_agent trained against random agent slightly worse (not by much)
Conclusion: rl not ideal for this type of play
Note: rl conclusions based on code running in my kaggle environment, since I've not got it ported over here yet to to Python version issues

Callout to:
* Noah Fang - for the idea and I also used his idea for how to check for n-in-a-row
* Kaggle - the playing agents were derived from https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning
* Acamol - the ConnectFourBoard control was derived from code in https://github.com/Acamol/connect-four





