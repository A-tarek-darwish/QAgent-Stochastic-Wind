This file contains a Python code for a SARSA Agent that navigates a maze that contains immobile obstacles and stochastic wind. It is initially set to run for a maximum of 500 episodes. It also contains a GUI to show the optimal policy developed by the agent and a graph showing the training performance across episodes. 


Grid and Reward Information:

• The grid world size is 6x6.
• The environment is undiscounted (i.e. 𝛾 = 1).
• The goal should be at the bottom right corner.
• There are obstacles in the environment as shown in the above figure at the following positions (1, 2), (1, 3), (1, 4), (4, 2), (4, 3) and (4, 4).
• The allowable actions are “UP”, “DOWN”, “LEFT” and “RIGHT”.
• The wind acts upwards in the 2 middle columns of the environment; such that there is a 70% probability of the agent being pushed upwards regardless of the action taken.
• The rewards are described as follows. Reaching the goal position is rewarded by 20 units. Taking an action against the wind direction is penalized by 5 units as it consumes the agent’s energy. Also, colliding with an obstacle results in a penalty of 5 units. The agent should be motivated to reach the goal as soon as possible; so, each time step is otherwise penalized by 1 unit.
