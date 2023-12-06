# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:48:12 2023

@author: jim

Windy Grid World(Stochastic)(King's move)
"""
import numpy as np
import random
import time

class GridMap:
    def __init__(self, select=0):    
        selectmap = [self.draw_map_1]
        b, w, e, s = selectmap[select]()
        self.start_location = s
        self.blank_map = b
        self.wind_map = w
        self.end_location = e
        
    def draw_map_1(self):
        b_map = np.zeros((10,7))
        w_map = np.array([0,0,0,1,1,1,2,2,1,0])
        e_loc = np.array([7,3])
        s_loc = np.array([0,3])
        return b_map, w_map, e_loc, s_loc


class GridWorld:
    def __init__(self, policy=0):
        grid_map = GridMap(0)
        self.select_map = 0
        self.blank_map = grid_map.blank_map
        self.x_range = self.blank_map.shape[0]
        self.y_range = self.blank_map.shape[1]
        self.wind_map = grid_map.wind_map
        self.end_location = grid_map.end_location
        self.current_location = grid_map.start_location
        self.valid_move = np.array([[1,1],[0,1],[-1,1],[1,0],[-1,0],[1,-1],[0,-1],[-1,-1]])
        self.Q = np.zeros((self.x_range,self.y_range,len(self.valid_move)))
        self.alpha = 0.1
        self.reward = -1
        self.epsilon = 0.1
        self.V = np.zeros((self.x_range,self.y_range))
        self.policy = [self.n_step_Sarsa]
        self.selected_policy = self.policy[policy]
        
    def redirect_location(self, current_location):
        if current_location[0]<0 : current_location[0]=0
        if current_location[1]<0 : current_location[1]=0
        if current_location[0]>=self.x_range : current_location[0] = self.x_range-1
        if current_location[1]>=self.y_range : current_location[1] = self.y_range-1
        return current_location
    
    def move(self, move_direction):
        current_location = self.current_location + self.valid_move[move_direction]
        current_location = self.redirect_location(current_location)
        current_location = self.stochastic_wind(current_location)
        current_location = self.redirect_location(current_location)
        return current_location
    
    def stochastic_wind(self, location):
        if self.wind_map[location[0]] == 0 : return location
        else : return location + np.array([0,random.randint(-1, 1) - self.wind_map[location[0]]])
    
    def epsilon_greedy(self, location, e):
        rdfp = random.random()
        if rdfp>e:
            location, max_a = self.greedy(location)
        else:
            rd = random.randint(0, 7)
            max_a = rd 
        return location, max_a
    
    def greedy(self, location):
        max_a = 0
        max_q = -1000000
        policy_with_same_value = []
        for i in range(0,8):
            if max_q == self.Q[location[0]][location[1]][i]:
                policy_with_same_value.append(i)
            elif max_q < self.Q[location[0]][location[1]][i]:
                max_a = i
                max_q = self.Q[location[0]][location[1]][i]
                policy_with_same_value = [i]
        if len(policy_with_same_value) > 1:
            rd = random.randint(0, len(policy_with_same_value)-1)
            max_a = policy_with_same_value[rd]
        return location, max_a # S, A | max_a returns a int

    
    def fetch_Q(self, s, a):
        return self.Q[s[0]][s[1]][a]
    
    def n_step_Sarsa(self, n=2):
        grid_map = GridMap(self.select_map)
        history = []
        T = 10000000
        t = 0
        tau = 0
        s, a=self.epsilon_greedy(self.current_location, self.epsilon)
        history.append((s, a))
        while not (self.current_location == self.end_location).all() or tau != T-1 :
            if t < T:
                self.current_location = self.move(a) # reward = -1
                s_new = self.current_location
                if (s_new == self.end_location).all():
                    T = t + 1
                else:
                    s_new, a=self.epsilon_greedy(self.current_location, self.epsilon)
                    history.append((s_new, a))
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau + n, T)+1):
                    G += -1.0 # gamma * Ri
                if tau + n < T:
                    G += self.fetch_Q(history[tau+n][0], history[tau+n][1])
                # Q(tau) = Q(tau)+ alpha*[G - Q(tau)]
                self.Q[history[tau][0][0]][history[tau][0][1]][history[tau][1]] += self.alpha * (G - self.fetch_Q(history[tau][0], history[tau][1]))
            t += 1
        self.current_location = grid_map.start_location

    def run_eval(self):
        grid_map = GridMap(self.select_map)
        count = 0
        while not (self.current_location == self.end_location).all():
            s, a = self.greedy(self.current_location)
            self.current_location = self.move(a)                 
            # time.sleep(0.1)
            count+=1
            if count > 1000:
                self.current_location = grid_map.start_location # infinite loop fail safe
        self.current_location = grid_map.start_location
        return count
    
    def iterate(self, iteration):
        for i in range(0,iteration):
            self.selected_policy()
        
print("--------------------------")
print("n_step_Sarsa:")
n_Sarsa = GridWorld(0)
n_Sarsa.iterate(20000)
print("training done")
acc = 0
for i in range (0,100): 
    acc+=n_Sarsa.run_eval()
print (acc/100)
print("--------------------------")
