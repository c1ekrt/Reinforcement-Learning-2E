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
        self.current_location  = grid_map.start_location
        self.valid_move = np.array([[1,1],[0,1],[-1,1],[1,0],[-1,0],[1,-1],[0,-1],[-1,-1]])
        self.Q = np.zeros((self.x_range,self.y_range,len(self.valid_move)))
        self.old_Q = np.zeros((self.x_range,self.y_range,len(self.valid_move)))
        self.alpha = 0.5
        self.reward = -1
        self.epsilon = 0.1
        self.V = np.zeros((self.x_range,self.y_range))
        self.policy = [self.run_Sarsa_epsilon_greedy, self.run_Q_learning_epsilon_greedy, self.run_Expected_Sarsa_epsilon_greedy]
        self.selected_policy = self.policy[policy]
        
    def redirect_location(self, current_location):
        if current_location[0]<0 : current_location[0]=0
        if current_location[1]<0 : current_location[1]=0
        if current_location[0]>=self.x_range : current_location[0] = self.x_range-1
        if current_location[1]>=self.y_range : current_location[1] = self.y_range-1
        return current_location
    
    def move(self, move_direction):
        current_location =self.current_location + self.valid_move[move_direction]
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
            if max_q == self.old_Q[location[0]][location[1]][i]:
                policy_with_same_value.append(i)
            elif max_q < self.old_Q[location[0]][location[1]][i]:
                max_a = i
                max_q = self.old_Q[location[0]][location[1]][i]
                policy_with_same_value = [i]
        if len(policy_with_same_value) > 1:
            rd = random.randint(0, len(policy_with_same_value)-1)
            max_a = policy_with_same_value[rd]
        return location, max_a # S, A | max_a returns a int
    
    def Expect_Sarsa_greedy(self, location):
        max_a = 0
        max_q = -1000000
        policy_with_same_value = []
        for i in range(0,8):
            if max_q == self.old_Q[location[0]][location[1]][i]:
                policy_with_same_value.append(i)
            elif max_q < self.old_Q[location[0]][location[1]][i]:
                max_a = i
                max_q = self.old_Q[location[0]][location[1]][i]
                policy_with_same_value = [i]
        if len(policy_with_same_value) > 1:
            rd = random.randint(0, len(policy_with_same_value)-1)
            max_a = policy_with_same_value[rd]
        return location, max_a # S, A | max_a returns a int
    
    def Expected_Sarsa_calculate_expectation(self, location):
        max_a = 0
        max_q = -1000000
        policy_with_same_value = []
        for i in range(0,8):
            if max_q == self.old_Q[location[0]][location[1]][i]:
                policy_with_same_value.append(i)
            elif max_q < self.old_Q[location[0]][location[1]][i]:
                max_a = i
                max_q = self.old_Q[location[0]][location[1]][i]
                policy_with_same_value = [i]
        non_optimal_action_prob = self.epsilon/8 # 8 is action count
        optimal_action_prob = (1.0-self.epsilon)/len(policy_with_same_value) + non_optimal_action_prob
        # calculate Expectation
        E = 0
        policy_with_same_value_set = set(policy_with_same_value)
        for i in range(0,8):
            if i in policy_with_same_value_set:
                E += optimal_action_prob * self.old_Q[location[0]][location[1]][i]
            else :
                E += non_optimal_action_prob * self.old_Q[location[0]][location[1]][i]
        return E
    
    def fetch_Q(self, s, a):
        return self.old_Q[s[0]][s[1]][a]
    
        
    def run_Sarsa_epsilon_greedy(self):
        grid_map = GridMap(self.select_map)
        self.Q = self.old_Q.copy()
        s, a = self.epsilon_greedy(self.current_location, self.epsilon)
        while not (self.current_location == self.end_location).all():
            self.current_location = self.move(a)                    
            s_new, a_new = self.greedy(self.current_location)  # S', A'
            self.old_Q[s[0]][s[1]][a] += self.alpha * (-1 + self.fetch_Q(s_new, a_new)-self.fetch_Q(s, a))
            s = s_new
            a = a_new
        self.current_location = grid_map.start_location
        
    def run_Q_learning_epsilon_greedy(self):
        grid_map = GridMap(self.select_map)
        self.Q = self.old_Q.copy()
        while not (self.current_location == self.end_location).all():
            s, a = self.epsilon_greedy(self.current_location, self.epsilon)  # S, A
            self.current_location = self.move(a)
            s_new, a_max = self.greedy(self.current_location)
            self.old_Q[s[0]][s[1]][a] += self.alpha * (-1 + self.fetch_Q(s_new, a_max)-self.fetch_Q(s, a))
            s = s_new
        self.current_location = grid_map.start_location
        
    def run_Expected_Sarsa_epsilon_greedy(self):
        grid_map = GridMap(self.select_map)
        self.Q = self.old_Q.copy()
        while not (self.current_location == self.end_location).all():
            s, a = self.epsilon_greedy(self.current_location, self.epsilon)  # S, A
            self.current_location = self.move(a)
            s_new, a_max = self.greedy(self.current_location)
            s_new_action_expectation = []
            for i in range(0, 8): # compare Expectation
                s_new_action_expectation.append(self.Expected_Sarsa_calculate_expectation(s_new))
            max_expectation = max(s_new_action_expectation)
            self.old_Q[s[0]][s[1]][a] += self.alpha * (-1 + max_expectation-self.fetch_Q(s, a))
            s = s_new
        self.current_location = grid_map.start_location
        
    def run_eval(self):
        grid_map = GridMap(self.select_map)
        count = 0
        while not (self.current_location == self.end_location).all():
            s, a = self.greedy(self.current_location)
            self.current_location = self.move(a)   
            # print(s)                
            # time.sleep(0.1)
            count+=1
        self.current_location = grid_map.start_location
        return count
    
    def iterate(self, iteration):
        for i in range(0,iteration):
            self.selected_policy()
        # print(self.Q)
        
print("--------------------------")
print("Sarsa:")
Sarsa = GridWorld(0)
Sarsa.iterate(10000)
print("training done")
acc = 0
for i in range (0,100): 
    acc+=Sarsa.run_eval()
print (acc/100)
print("--------------------------")
print("Q-learning:")
Q = GridWorld(1)
Q.iterate(10000)
print("training done")
acc = 0
for i in range (0,100):
    acc+=Q.run_eval()
print (acc/100)
print("--------------------------")
print("Expected_Sarsa:")
Expected_Sarsa = GridWorld(2)
Expected_Sarsa.iterate(10000)
print("training done")
acc = 0
for i in range (0,100):
    acc+=Expected_Sarsa.run_eval()
print (acc/100)
