# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:43:30 2023

@author: jim
"""
import numpy as np
import matplotlib.pyplot as plt

class GamblersProblem:
    def __init__(self, terminal_state=100, win_prob=0.40):
        '''
        terminal state:(uint)
            The winning condition, which is at how many coins can you get the reward.
        win_prob:(float32 in region of (0,1))
            Probability of winning the money.
            If you win, you got your money in addition to same amount of money you bet as reward.
            If you lose, you lose all the money you bet and got nothing.
        '''
        self.state_value_graph = np.zeros(terminal_state, dtype=float)
        self.a_graph = np.zeros((terminal_state),dtype=int)
        self.terminal_state = terminal_state
        self.win_prob = win_prob
        self.lose_prob = 1.0 - win_prob
        self.win_reward = 1.0
        self.old_state_value_graph = np.array(self.state_value_graph, dtype=float)
    def reward(self, state):
        if state >= self.terminal_state:
            return self.win_reward

        return self.old_state_value_graph[state]
    
    def calculate_value(self, state):
        max_reward = 0.0
        action = 0
        # bet larger than upper limit is trivial since we always got reward 1 even with over {terminal_state} dollars in pocket. Why risk more for nothing :D
        upperlimit = min(self.terminal_state - state, state) 
        for s in range(1, upperlimit + 1):
            state_value = self.win_prob * self.reward(state+s) + self.lose_prob * self.reward(state-s)
            if max_reward < state_value:
                action = s
                max_reward = state_value
        return action, max_reward
    
    def print_everything(self):
        data = np.arange(0, self.terminal_state)
        plt.subplot(221)
        plt.title("state_value")
        plt.xlabel("State") # x
        plt.ylabel("Expectation") # y
        plt.bar(data, self.state_value_graph)
        plt.subplot(222)
        plt.title("action")
        plt.xlabel("State") # x
        plt.ylabel("Action") # y
        plt.bar(data, self.a_graph)
        plt.show()
        
    def iteration(self, step):
        for iteration in range(0, step):
            print("-----------------------------------------------------------------------")
            print(iteration)
            for s in range(1, self.terminal_state):
                action, max_reward = self.calculate_value(s)
                self.state_value_graph[s] = max_reward
                self.a_graph[s] = action
            self.old_state_value_graph=np.array(self.state_value_graph)
            
def __main__():
    g = GamblersProblem()
    g.iteration(1000)
    g.print_everything()
    # g.iteration(1)
    # g.print_everything()

__main__()