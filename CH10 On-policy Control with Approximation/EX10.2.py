# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 08:10:23 2023

@author: jim
"""

import numpy as np
import random
import time

class AccessControlQingTask:
    def __init__(self):
        self.available_server = 10
        self.customer_priority = [1,2,4,8]
        self.w = np.zeros((11,4,2), dtype=float)
        self.free_prob = 0.06
        self.step_count = 0
        self.reward_avg = 0
        self.alpha = 0.01
        self.beta = 0.01
        self.epsilon = 0.1
        self.count = np.zeros((11,4,2), dtype=int)
        
    def timestep_free_server(self):
        server = self.available_server
        for i in range (0, 10-server):
            rd = random.random()
            if rd <= self.free_prob:
                self.available_server += 1
                
    def generate_customer(self):
        rd = random.randint(0, 3)
        return rd
    
    def epsilon_greedy(self, s):
        if s[0] == 0:
            return 0
        rd = random.random()
        if rd <= self.epsilon:
            rd = random.randint(0, 1)
            return rd
        else:  
            return self.greedy(s)
            
    def greedy(self, s): 
        if s[0] == 0:
            return 0
        if self.w[s[0]][s[1]][0] == self.w[s[0]][s[1]][1]:
            return random.randint(0, 1)
        elif self.w[s[0]][s[1]][0] > self.w[s[0]][s[1]][1]:
            return 0
        else:
            return 1
            
    def fetch_q_s_a(self, s, a):
        if s[0] < 0:
            return 0
        return self.w[s[0]][s[1]][a]
    
    def diff_semi_grad_Sarsa(self, iteration):
        customer = self.generate_customer()
        # initialize S, A
        s = (self.available_server, customer)
        a = random.randint(0, 1)    # 1:accept, 0,reject
        # Loop for each steps
        while self.step_count < iteration:
            self.count[s[0]][s[1]][a] += 1
            self.timestep_free_server()
            self.step_count += 1
            
            # Take action A observer R S'
            if a == 1:
                r = self.customer_priority[s[1]]
                self.available_server -= 1
            elif a == 0:
                r = 0
                # do nothing
                
            # Choose A' as a function of q   
            customer = self.generate_customer()
            new_s = (self.available_server, customer)
            new_a = self.epsilon_greedy(new_s)
            
            # delta calc
            delta = r - self.reward_avg + self.fetch_q_s_a(new_s, new_a) - self.fetch_q_s_a(s, a)
            
            # R bar update
            self.reward_avg += self.beta * delta
            
            # weight update
            self.w[s[0]][s[1]][a] += self.alpha * delta * 1
            
            s = new_s
            a = new_a
                         
    def run_eval(self, s):
        return self.greedy(s)
        
    def graph(self):
        print ("   1  2  4  8")
        for server in range(0, 11):
            g = []
            print(server, end=" ")
            for customer in range(0,4):
                g.append(self.greedy((server,customer)))
            print(g)
        print("Reject")
        print(self.w[:,:,0])
        print("Granted")
        print(self.w[:,:,1])
        print(self.count[:,:,0])
        print(self.count[:,:,1])

ACQT = AccessControlQingTask()
ACQT.diff_semi_grad_Sarsa(1000000)
np.set_printoptions(precision=4, suppress=True)
ACQT.graph()




