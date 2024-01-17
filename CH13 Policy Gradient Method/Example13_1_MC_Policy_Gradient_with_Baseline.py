# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:54:42 2024

@author: jim
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math

class ShortCorridorBias:
    def __init__(self):
        self.action = [0,1]
        self.state_direction = [[0,1],[1,-1],[-1,1]]
        self.x_s_left = [0,1]
        self.x_s_right = [1,0]
        self.alpha_w = 0.0002
        self.alpha_theta=0.0008
        self.theta = np.zeros((2), dtype=float)
        self.weight = np.zeros((4), dtype=float)
        self.prob = [0.99, 0.01]
    
    def x_s_a_minus_sum_of_prob_times_x_s_b(self, s, a):
        x_s = 0
        sum_of_prob_times_x_s_b = np.zeros((2))
        # x(s,a)
        if a == 0: 
            x_s = self.x_s_left
        else: 
            x_s = self.x_s_right
        # sum_of_prob_times_x_s_b
        sum_of_prob_times_x_s_b += (np.array(self.x_s_left) * self.prob[0])
        sum_of_prob_times_x_s_b += (np.array(self.x_s_right) * self.prob[1])
        x_s -= sum_of_prob_times_x_s_b
        return x_s
        
    # pnp stands for parameterized numerical preference, h as notation
    def pnp(self, s, a):
        x_s_a = []
        if a == 1:
            x_s_a = self.x_s_right
        else:
            x_s_a = self.x_s_left
        value = np.dot(self.theta, x_s_a)
        return value
        
    # you need sum of pnp of all action in order to calculate softmax
    def sum_of_pnp(self, s):
        return self.pnp(s, 0) + self.pnp(s, 1)
    
    def softmax(self, s, a):
        return math.exp(self.pnp(s,a))/math.exp(self.sum_of_pnp(s))
        
    def action_prob(self, s):
        self.prob[0] = self.softmax(s,0)
        self.prob[1] = 1.0 - self.prob[0]
        
    def next_action(self):
        rd = random.random()
        if rd < self.prob[0]:
            return 0
        else:
            return 1
        
    def generate(self):
        # (S0 A0 R1) (S1 A1 R2)
        history = []
        s = 0
        reward = 0
        while s != 3:
            a = self.next_action()
            r_new = -1
            history.append((s,a,r_new))
            s += self.state_direction[s][a]
            reward += r_new
        return history, reward
            
    def evaluate(self):
        ttr = 1000
        step = 0
        state = 0
        for i in range(0, ttr):
            while state != 3:
                step -= 1
                rd = random.random()
                if rd > self.prob[0]:
                    state += self.state_direction[state][0]
                else:
                    state += self.state_direction[state][1]
            state = 0
        avg = step/float(ttr)
        return avg
        
    def draw(self, avg, episode):
        x = np.arange(episode)
        y = avg
        plt.plot(x, y,color="green", label='MCPG_with_Baseline')
        
    def run(self, episode):
        all_avg = []
        for i in range (0, episode):
            history, total_reward= self.generate()
            G = total_reward
            for h in history:
                s_t, a_t, r_t = h   # r_t is r_{t+1} here
                delta = G - self.weight[s_t]
                self.weight[s_t] = self.weight[s_t] + self.alpha_w * delta * 1 
                self.theta += self.alpha_theta * delta * (self.x_s_a_minus_sum_of_prob_times_x_s_b(s_t, a_t))
                
                G -= r_t
            self.action_prob(0)
            
            # this prevent probability over 1 or under zero (for some reason)
            if self.prob[0] > 1:
                self.prob = [0.99, 0.01]
                self.theta = np.zeros((2), dtype=float)
            if self.prob[0] < 0:
                self.prob = [0.01, 0.99]
                self.theta = np.zeros((2), dtype=float)
                
            all_avg.append(self.evaluate())
        print(all_avg)
        self.draw(all_avg, episode)
        
# sc = ShortCorridorBias()
# sc.run(1000)