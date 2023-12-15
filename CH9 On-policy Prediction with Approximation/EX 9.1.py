# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 07:58:04 2023

@author: jim

1000-state Random Walk (Gradient MC and semi-gradient TD) with 9.4 linear methods and constant alpha

"""
import numpy as np
import random 
import matplotlib.pyplot as plt
import time

class RandomWalk:
    def __init__(self):
        self.terminal_upperbond = 1000
        self.terminal_lowerbound = 1
        self.upperbond_reward = 1.0
        self.lowerbond_reward = -1.0
        self.s = 500
        self.partition_count = 10
        self.w = np.zeros((self.partition_count))
        self.v_hat = np.zeros((self.partition_count))
        self.alpha = 0.0002
        
        
    def detect_terminal(self, s):
        if s < self.terminal_lowerbound:
            return self.lowerbond_reward
        if s > self.terminal_upperbond:
            return self.upperbond_reward
        return 0.0
    
    def next_state(self, s):
        next_diff = random.randint(-100, 100)
        s += next_diff
        result = self.detect_terminal(s)
        return result, s
    

    
    def x_s(self, s):
        v_hat = np.zeros((self.partition_count))
        for i, vh in enumerate(v_hat):
            v_hat[i] = ((self.terminal_upperbond/self.partition_count*i) - s)/1000 # 9.4 linear methods x(s)
        return v_hat
    
    def v_hat_s_w(self, s):
        x_s = self.x_s(s)
        v_hat_sum = 0
        if self.detect_terminal(s) != 0:
            return 0.0
        for i, x_element in enumerate(x_s):
            v_hat_sum += x_element * self.w[i]
        return v_hat_sum
    
    '''==============================================MC=SGD======================================================'''
    
    def run_MC_SGD(self):
        hist = []
        reward = 0
        hist.append(self.s)
        while True:
            reward, new_s = self.next_state(self.s)
            if reward != 0.0:
                break
            else: hist.append(new_s)
            self.s = new_s
        for h in hist:
            self.w += (self.alpha * (reward - self.v_hat_s_w(h)) * self.x_s(h))
            # print(self.w)
        self.s = 500
            
    def iterate_MC_SGD(self, n):
        self.__init__()
        for i in range (0, n):
            self.run_MC_SGD()
            # print("w", self.w)
            if i % 1000 == 0:
                print(i)
            
    def test_MC_SGD(self):
        state = []
        v_hat = []
        for i in range(1, self.partition_count):
            state.append(self.terminal_upperbond/self.partition_count*i)
            v_hat.append(self.v_hat_s_w((self.terminal_upperbond/self.partition_count*i)))
        print(state)
        print(v_hat)
        plt.bar(state, v_hat, width=80)
        # plt.xlabel("state")
        # plt.ylabel("v_hat")
        # plt.legend()
        plt.show()
    
    '''==============================================Semi=TD======================================================'''
    
    def run_semi_TD0(self):
        self.s = 500
        reward = 0
        while True:
            reward, new_s = self.next_state(self.s)
            self.w += self.alpha * (reward + self.v_hat_s_w(new_s) - self.v_hat_s_w(self.s)) * self.x_s(self.s)
            self.s = new_s
            if reward != 0.0:
                break
            
    def iterate_semi_TD0(self, n):
        self.__init__()
        for i in range (0, n):
            self.run_semi_TD0()
            # print("w", self.w)
            if i % 1000 == 0:
                print(i)
            
    def test_semi_TD0(self):
        state = []
        v_hat = []
        for i in range(1, self.partition_count):
            state.append(self.terminal_upperbond/self.partition_count*i)
            v_hat.append(self.v_hat_s_w((self.terminal_upperbond/self.partition_count*i)))
        print(state)
        print(v_hat)
        plt.bar(state, v_hat, width=80)
        # plt.xlabel("state")
        # plt.ylabel("v_hat")
        # plt.legend()
        plt.show()
        
'''
Gradient Monte Carlo Algorithm
'''
rdw = RandomWalk()
rdw.iterate_MC_SGD(10000)
rdw.test_MC_SGD()

'''
Semi-gradient TD(0)
'''
# sgtd = RandomWalk()
# sgtd.iterate_semi_TD0(10000)
# sgtd.test_semi_TD0()
