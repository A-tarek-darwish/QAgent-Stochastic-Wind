# Creator: Abdelrahman Darwish 
# GUC ID: 40-4072
# Date : 30/4/2021

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import time
import random

UNIT = 100 # pixels 7

class Environment:
    def __init__(self, wgw_cols=4, wgw_rows=4, treasure_pos=(0, 0), wind_pos=[], obstacles_pos=[], delta_t=0.3):
        self.n_states = (wgw_rows , wgw_cols)
        self.actions_set = ["U", "D", "L", "R"]
        self.dT = delta_t
        self.curr_state = None
        self.curr_action = None
        self.treasure_pos = treasure_pos
        self.wind_pos = wind_pos
        self.obstacles_pos = obstacles_pos
        self.wgw_window = tk.Tk()
        self.wgw_canvas = tk.Canvas(self.wgw_window , bg="white", 
                    height=self.n_states[0] * UNIT, width=self.n_states[1] * UNIT)
        self.explorer = None
        self.action_arrow = None
        self.build_wgw_window()

    """
    This builds the GUI using tkinter for the windy grid environment.
    The x-axis is positive to the right.
    The y-axis is positive downwards.
    The winds are shown in blue arrows.
    The obstacles are shown in yellow boxes.
    The goal position is green circle.
    self.explorer is the agent itself (red rectangle).
    The optimal policy is shown in black arrows.
    """

    def build_wgw_window(self):
        # create grids: horizontal lines
        for row in range(0, self.n_states[0] * UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, self.n_states[0] * UNIT, row
            self.wgw_canvas.create_line(x0, y0, x1, y1, width=2, fill = ('black'))
        
        # create grids: vertical lines
        for col in range(0, self.n_states[1] * UNIT, UNIT):
            x0, y0, x1, y1 = col, 0, col, self.n_states[1] * UNIT
            self.wgw_canvas.create_line(x0, y0, x1, y1, width=2, fill =('black'))

        # create origin
        origin = np.array([0.5 * UNIT, 0.5 * UNIT])

        # create obstacles
        for i in range(len(self.obstacles_pos)):
            obs_center = np.array(self.obstacles_pos[i]) * UNIT + 0.5 * UNIT
            self.wgw_canvas.create_rectangle(obs_center[0] - 15, obs_center[1] - 15,obs_center[0] + 15, obs_center[1] + 15, fill='yellow')

         # create wind
        self.wgw_canvas.create_line( 350 , 50  , 350 ,1000 , width = 5,  arrow = 'first', fill= ('blue'))
        self.wgw_canvas.create_line( 250 , 50  , 250 ,1000 , width = 5,  arrow = 'first', fill= ('blue'))

        # create treasure
        oval_center = np.array(self.treasure_pos) * UNIT + UNIT * 0.5 
        self.wgw_canvas.create_oval(oval_center[0] - 25, oval_center [1] 
            - 25, oval_center[0] + 25, oval_center[1] + 25, fill=('green'))

        # create explorer
        self.explorer = self.wgw_canvas.create_rectangle(origin[0] - 
            15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill=('red'))
        # pack all
        self.wgw_canvas.pack()
        self.wgw_window.title('Windy Grid World') 
        self.wgw_window.geometry('{0}x{1}'.format(self.n_states[0] * UNIT, self.n_states[1] * UNIT))

    def update_wgw_window(self, dx, dy, action=None): 
        time.sleep(self.dT)
        if action is not None:
            # draw arrow for action
            x0 = self.curr_state[0] * UNIT + 0.5 * UNIT 
            y0 = self.curr_state[1] * UNIT + 0.5 * UNIT 
            x1, y1 = x0, y0
            if action == "U":
                y1=y0-0.5* UNIT 
            elif action == "D":
                y1=y0+0.5* UNIT 
            elif action == "R":
                x1=x0+0.5* UNIT 
            elif action == "L":
                x1=x0-0.5* UNIT
            if self.action_arrow is not None:
                self.wgw_canvas.delete(self.action_arrow) 

            self.action_arrow = self.wgw_canvas.create_line(x0, y0, 
                x1, y1, arrow=tk.LAST, width=6, fill="black", arrowshape=(15, 15, 8))
            
            # move agent
        self.wgw_canvas.move(self.explorer, dx * UNIT, dy * UNIT) 
        self.wgw_canvas.update()
        self.wgw_window.update()
        

    def show_optimal_policy (self, q_table):
        range_random_wind = random.random()
        self.wgw_canvas.delete(self.explorer) 
        self.wgw_canvas.delete(self.action_arrow) 
        for xi in range(self.n_states[0]):
            for yi in range(self.n_states[1]):
                if (xi, yi) == self.treasure_pos or (xi, yi) in self.obstacles_pos : 
                     continue
                x0 = xi * UNIT + 0.5 * UNIT 
                y0 = yi * UNIT + 0.5 * UNIT 
                all_idx = np.argwhere(np.ndarray.flatten(q_table[xi, yi, :]) == np.amax(q_table[xi, yi, :]))
                all_idx = np.ndarray.flatten(all_idx)
                for ai in all_idx:
                    x1, y1 = x0, y0
                    if (xi, yi) in self.wind_pos:
                        if self.actions_set[ai] == "U":
                            y1 = y0 - 0.5 * UNIT
                        elif self.actions_set[ai] == "D":
                            if range_random_wind >=0.7:
                                y1 = y0 - 0.5 * UNIT
                            else:
                                y1 = y0 + 0.5 * UNIT
                        elif self.actions_set[ai] == "R":
                            if range_random_wind >=0.7:
                                y1 = y0 - 0.5 * UNIT
                            else:
                                x1 = x0 + 0.5 * UNIT
                        elif self.actions_set[ai] == "L":
                            if range_random_wind >=0.7:
                                y1 = y0 - 0.5 * UNIT
                            else:
                                x1 = x0 - 0.5 * UNIT
                    if self.actions_set[ai] == "U": 
                        y1 = y0 - 0.5 * UNIT
                    elif self.actions_set[ai] == "D": 
                        y1 = y0 + 0.5 * UNIT
                    elif self.actions_set[ai] == "R": 
                        x1 = x0 + 0.5 * UNIT
                    elif self.actions_set[ai] == "L": 
                        x1 = x0 - 0.5 * UNIT
                self.wgw_canvas.create_line(x0, y0, x1, y1, arrow=tk.LAST, 
                    width=6, fill="black", arrowshape=(15, 15, 8))

        self.wgw_canvas.update() 
        self.wgw_window.update()

    def env_start(self, init_state): 
        self.curr_state = init_state
        if self.action_arrow is not None:
            self.wgw_canvas.delete(self.action_arrow) 
            self.wgw_canvas.delete(self.explorer)
            self.explorer = self.wgw_canvas.create_rectangle(0.5*UNIT -
                15, 0.5*UNIT - 15, 0.5*UNIT + 15, 0.5*UNIT + 15, fill='red')
            self.update_wgw_window(dx=init_state[0], dy=init_state[1])

    def env_step(self, action):
        new_x, new_y = self.curr_state
        range_random_wind = random.random()
        if action == "U":
            new_y = max(0, self.curr_state[1]-1)
        elif action == "D":
            new_y = min(self.curr_state[1] + 1, self.n_states[1]-1)
        elif action =="R":
            new_x = min(self.curr_state[0]+1 , self.n_states[0]-1)
        elif action =="L":
            new_x = max(0, self.curr_state[0]-1)
        reward = -1
        
        if (new_x, new_y) in self.obstacles_pos:
            reward = -5
            new_x, new_y = self.curr_state

        if (new_x, new_y) in self.wind_pos: 
            if action == "U":
                new_y = max(0, self.curr_state[1]-1)
                reward = -1
            elif action == "D":
                if range_random_wind >=0.7:
                    new_y = max(0, self.curr_state[1]-1)
                    reward = -5
                else:
                    new_y = min(self.curr_state[1] + 1, self.n_states[1]-1)
                    reward = -5
            elif action == "R":
                if range_random_wind >=0.7:
                    new_y = max(0, self.curr_state[1]-1)
                    reward = -5
                else:
                    new_x = min(self.curr_state[0]+1 , self.n_states[0]-1)
                    reward = -5
            elif action == "L":
                if range_random_wind >=0.7:
                    new_y = max(0, self.curr_state[1]-1)
                    reward = -5
                else:
                    new_x = max(0, self.curr_state[0]-1)
                    reward = -5
        self.update_wgw_window(new_x-self.curr_state[0], new_y-self.curr_state[1], action)
        self.curr_state = (new_x, new_y) 
        self.curr_action = action

        # check for terminal state
        terminal = self.curr_state == self.treasure_pos 
        if terminal:
            reward = 20
        return [terminal , reward]


################################################
def my_argmax(arr):
    all_idx = np.argwhere(np.ndarray.flatten(arr) == np.amax(arr))
    return np.random.choice(np.ndarray.flatten(all_idx))
    
class SARSA_Agent():
    
    # Initialize environment, gamma, alpha, epsilon, max episodes and reward
    def __init__(self, environment, epsilon,gamma, alpha, max_episodes):
        self.env = environment
        self.discount = gamma  
        self.lr = alpha
        self.eps = epsilon
        self.max_episodes = max_episodes 
        self.q_table = np.zeros(shape=(self.env.n_states[0], self.env.n_states[1], len(self.env.actions_set)))
        self.prev_state = None
        self.prev_action = None
        

    def e_greedy_policy(self, state):
            if np.random.uniform() < self.eps:
                action_selected = np.random.choice(len(self.env.actions_set))
            else:
                action_selected = my_argmax(self.q_table[state[0], state[1], :])
            return action_selected

    def agent_start(self, init_state):
        self.env.env_start(init_state)
        self.prev_state = init_state
        self.prev_action = self.e_greedy_policy(init_state)
        [terminal, reward] = self.env.env_step(self.env.actions_set[self.prev_action])
        return [terminal, reward]

    def agent_step(self, reward):
        curr_state = self.env.curr_state
        curr_action = self.e_greedy_policy(curr_state)
        q_target = reward + self.discount * np.max(self.q_table[curr_state[0], curr_state[1], curr_action]) #Difference between a SARSA Agent and a Q Agent
        q_predict = self.q_table[self.prev_state[0], self.prev_state[1], self.prev_action]
        self.q_table[self.prev_state[0], self.prev_state[1], self.prev_action] += \
            self.lr * (q_target - q_predict)
        self.prev_state = curr_state
        self.prev_action = curr_action
        [terminal, reward] = self.env.env_step(self.env.actions_set[self.prev_action])
        return [terminal, reward]

    def agent_end(self, reward):
        curr_state = self.env.curr_state
        q_predict = self.q_table[self.prev_state[0], self.prev_state[1], self.prev_action]
        q_target = reward
        self.q_table[self.prev_state[0], self.prev_state[1], self.prev_action] += \
            self.lr * (q_target - q_predict)
        self.prev_state = curr_state


if __name__ == "__main__":
    wgw = Environment(wgw_cols=6, wgw_rows=6,
                       treasure_pos=(5, 5),  wind_pos=[(2, 1), (3, 1), (1, 1), (1, 2), (1, 3)], 
                       obstacles_pos =[(1,2),(1,3),(1,4),(4,2),(4,3),(4,4)],
                       delta_t=0.1)
    agent = Q_Agent(environment=wgw, epsilon=0.05, gamma=0.999, alpha=0.1, max_episodes=500)
    returns = np.zeros(shape=(agent.max_episodes,))
    for e in range(agent.max_episodes):
        wgw.wgw_window.title("Windy Grid World - Episode #"+str(e+1))
        init_x = np.random.randint(0, wgw.n_states[0])
        init_y = np.random.randint(0, wgw.n_states[1])
        init_state = (init_x, init_y)
        if init_state in wgw.wind_pos:
            init_state = (0, 0)
        [terminal, reward] = agent.agent_start(init_state)
        returns[e] += reward
        i_step = 0
        while not terminal:
            [terminal, reward] = agent.agent_step(reward)
            i_step += 1
            returns[e] += reward  #* (0.9**i_step)
        agent.agent_end(reward)
    wgw.wgw_window.title("Windy Grid World - Optimal Policy Reached")
    wgw.show_optimal_policy(agent.q_table)
    plt.figure()
    plt.plot(np.arange(1, agent.max_episodes + 1, 1), returns)
    plt.xlabel("Episodes")
    plt.ylabel("Total Returns")
    plt.grid(True)
    plt.show()
