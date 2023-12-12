# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:27:15 2023

@author: jim

Tabular Dyna-Q with Dyna Maze
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class MazeMap:
    def __init__(self, map_choose):
        self.maze_map = [self.map_0]
        self.map_choose = map_choose
        self.start_location = np.array([0,0])
        self.end_location = np.array([0,0])
        
    def map_0(self):
        '''
        0 = space
        1 = wall
        8 = start location
        9 = end location
        '''
        map_layout = np.zeros((9,6))
        map_wall = [[2,2],[2,3],[2,4],[5,1],[7,3],[7,4],[7,5]]
        for loc in map_wall:
            map_layout[loc[0]][loc[1]] = 1
        self.start_location = np.array([0,3])
        map_layout[self.start_location[0]][self.start_location[1]] = 8
        self.end_location = np.array([8,5])
        map_layout[self.end_location[0]][self.end_location[1]] = 9
        print(map_layout)
        return map_layout
    
class DynaMaze:
    def __init__(self, map_choose=0, n=10):
        m = MazeMap(map_choose)
        self.current_map = m.maze_map[m.map_choose]()
        self.Q = np.zeros((9,6,4))
        self.model = dict()
        self.start_location = m.start_location.copy()
        self.end_location = m.end_location.copy()
        self.direction = [[1,0],[0,1],[-1,0],[0,-1]]
        self.current_location = self.start_location.copy()
        self.x_range = 9
        self.y_range = 6
        self.alpha = 0.1
        self.epsilon = 0.1
        self.gamma = 0.95
        self.reward = 100
        self.n = n
        
    def move(self, location, direction):
        next_location = location + self.direction[direction]
        next_location = self.redirect_location(next_location, location)
        return next_location
    
    def redirect_location(self, current_location, last_location):
        if (current_location[0] < 0 or current_location[0] >= self.x_range or
            current_location[1] < 0 or current_location[1] >= self.y_range or
            self.current_map[current_location[0]][current_location[1]] == 1 ):
            current_location = last_location
        return current_location
    

    
    def fetch_Q(self, s, a):
        return self.Q[s[0]][s[1]][a]
    
    def fetch_model(self, s, a):
        return self.model[s[0]][s[1]][a]
    
    def R(self, location):
        if (location == self.end_location).all():
            return 1.0
        else:
            return 0.0
        
    def epsilon_greedy(self, location, e, target_policy):
        rdfp = random.random()
        if rdfp>e:
            location, max_a = self.greedy(location, target_policy)
        else:
            rd = random.randint(0, 3)
            max_a = rd 
        return location, max_a
    
    def greedy(self, location, target_policy):
        max_a = 0
        max_q = -1000000
        policy_with_same_value = []
        for i in range(0,4):
            if max_q == target_policy[location[0]][location[1]][i]:
                policy_with_same_value.append(i)
            elif max_q < target_policy[location[0]][location[1]][i]:
                max_a = i
                max_q = target_policy[location[0]][location[1]][i]
                policy_with_same_value = [i]
        if len(policy_with_same_value) > 1:
            rd = random.randint(0, len(policy_with_same_value)-1)
            max_a = policy_with_same_value[rd]
        return location, max_a # S, A | max_a returns a int
    
    def run_Dyna_Q(self):
        count = 0
        while not (self.current_location == self.end_location).all():
            count += 1
            S = self.current_location.copy()                       # (a)
            _, A = self.epsilon_greedy(S, self.epsilon, self.Q)     # (b)
            S_new = self.move(self.current_location , A)    # (c)
            R = self.R(S_new)
            self.current_location = S_new.copy()
            _, max_a = self.greedy(self.current_location, self.Q)
            self.Q[S[0]][S[1]][A] += self.alpha * (R + self.gamma * self.fetch_Q(S_new, max_a) - self.fetch_Q(S, A)) # (d)
            # Model(S,A) <- R,S'
            self.model[(tuple(S),A)]=(R,tuple(S_new)) # (e)
            for i in range(0, self.n):
                rand_SA, rand_RS_new = random.choice(list(self.model.items()))
                # print(rand_SA)
                # print(rand_RS_new)
                rand_S, rand_A = rand_SA[0],rand_SA[1]
                rand_R, rand_S_new = rand_RS_new[0], rand_RS_new[1]
                _, max_a = self.greedy(rand_S_new, self.Q)
                self.Q[rand_S[0]][rand_S[1]][rand_A] += self.alpha * (rand_R + self.gamma * self.fetch_Q(rand_S_new, max_a) - self.fetch_Q(rand_S, rand_A)) # (d)
        self.current_location = self.start_location.copy()
        return count
        
    def draw_plot(self, accumulated_reward, accumulated_timestep, n):
        plt.plot(accumulated_timestep, accumulated_reward,label=f"n={n}")
        
    def iterate(self, iteration):
        accumulated_reward = [0]
        sum_timestep = 0
        accumulated_timestep = [0]
        for i in range(0,iteration):
            timestep = self.run_Dyna_Q()
            sum_timestep += timestep
            accumulated_reward.append(i)
            accumulated_timestep.append(sum_timestep)
        self.draw_plot( accumulated_reward, accumulated_timestep, self.n)
        # np.set_printoptions(precision=3, suppress=True)
        # print(self.Q)
        # print(self.model)

# because the stochastic nature of the training progress, 
# we simply observe how fast the algorithm converge to the right answer.(the incline of the learning progress)
Dyna_Q = DynaMaze(map_choose=0, n=1)
Dyna_Q.iterate(5)
Dyna_Q = DynaMaze(map_choose=0, n=5)
Dyna_Q.iterate(5)
Dyna_Q = DynaMaze(map_choose=0, n=15)
Dyna_Q.iterate(5)
Dyna_Q = DynaMaze(map_choose=0, n=50)
Dyna_Q.iterate(5)
plt.legend()
plt.show()