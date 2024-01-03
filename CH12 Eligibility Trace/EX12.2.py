# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:47:16 2023

@author: jim

Mountain Car 1 Tiling n tile True online Sarsa(lambda)
TODO: fixing terminal q(terminal,a)
"""
import numpy as np
import random
import math
import pprint
import time

class MountainCar:
    def __init__(self, n=32):
        self.position_bond = [-1.2, 0.5]
        self.position_grid = (self.position_bond[1] - self.position_bond[0])/n
        self.velocity_bond = [-0.07, 0.07]
        self.velocity_grid = (self.velocity_bond[1] - self.velocity_bond[0])/n
        self.position = 0
        self.velocity = 0
        self.action_pool = [-1,0,1]
        self.n = n
        self.w = np.zeros(((n*n*4)+1), dtype=float) 
        self.protopoint = []
        self.offset_pos = self.position_grid / 4
        self.offset_v = self.velocity_grid / 4
        self.w = np.zeros(((n*n*4)+1), dtype=float) 
        self.old_q = 0
        
        self.decay_rate_lambda = 0.85
        self.alpha = 0.25
        self.epsilon = 0.01
        self.gamma = 1
        
    def next_position(self, pos, velocity):
        new_pos = pos + velocity
        new_pos = max(self.position_bond[0], min(new_pos, self.position_bond[1]))
        
        return new_pos
    
    def next_velocity(self, pos, velocity, action):
        new_velocity = velocity + 0.001 * self.action_pool[action] - 0.0025 * math.cos(3 * pos)
        new_velocity = max(self.velocity_bond[0], min(new_velocity, self.velocity_bond[1]))
        return new_velocity
        
    def reset(self):
        rand = random.uniform(-0.6, -0.4)
        self.position = rand
        self.velocity = 0
    
    def create_protopoint(self):
        protopoint = []
        for offs in range (0, 4):
            for i in range(0,self.n):
                for j in range(0,self.n):
                    protopoint.append((round(self.position_bond[0] + (i + 0.5) * self.position_grid + self.offset_pos * (i - offs), 5), 
                                       round(self.velocity_bond[0] + (j + 0.5) * self.velocity_grid + self.offset_v * (i - offs), 5)))
        self.protopoint = protopoint
            
    def prod_of_linear_matrix(self, matrix_A, matrix_B):
        npC = matrix_A * matrix_B
        value = 0
        for npc in npC:
            value += npc
            
        return value
    
    def in_range_of(self, protopoint, target):
        if (protopoint[0] < target[0] and protopoint[0] + self.position_grid > target[0] and
            protopoint[1] < target[1] and protopoint[1] + self.velocity_grid > target[1] ):
            return 1
        else: return 0
        
    def return_x_s_a(self, s, a): # s[0] = position, s[1] = velocity
        x_s_a = []
        for point in self.protopoint:
            x_s_a.append(self.in_range_of(point, s))
        x_s_a.append(a)
        return np.array(x_s_a, dtype=float)
    
    def greedy(self, s):
        max_a = 0
        max_q_hat = -1000000
        for a in range(0,3):
            q_hat = self.prod_of_linear_matrix(self.w, self.return_x_s_a(s, a))
            if max_q_hat <= q_hat:
                max_a = a
                max_q_hat = q_hat
        return max_a
    
    def epsilon_greedy(self, s):
        if self.epsilon > random.random():
            return random.randint(0, 2)
        else:
            return self.greedy(s)
        
    def true_online_Sarsa(self, iteration):
        for i in range(0, iteration):
            # initialize S
            self.reset()
            s = [self.position, self.velocity]
            a = self.epsilon_greedy(s)
            
            # Choose A approx policy or near greedily from S using w
            x = self.return_x_s_a(s, a)
            self.z = np.zeros(((self.n*self.n*4)+1), dtype=float) 
            self.old_q = 0
            
            # Loop for each step episode
            step = 0
            while self.position < 0.5:
                time.sleep(0.01)
                # take action A , observe R, S'
                r = -1
                self.position = self.next_position(self.position, self.velocity)
                self.velocity = self.next_velocity(self.position, self.velocity, a)
                if self.position < self.position_bond[0]:
                    self.reset()
        
                # Choose A' approx policy or greedily from S' using w
                new_s = [self.position, self.velocity]
                new_a = self.epsilon_greedy(new_s)
                
                # x' = x(S',A')
                new_x = self.return_x_s_a(new_s, new_a)
                
                # Q = w * x
                # Q' = w * x'
                q = self.prod_of_linear_matrix(self.w, x)
                new_q = self.prod_of_linear_matrix(self.w, new_x)
                
                # \delta = R + \gamma * Q' - Q
                delta = r + self.gamma * new_q - q
                
                # z = \gamma * \lambda * z + (1 - \alpha * \gamma * \lambda * z * x) * x
                self.z = (self.gamma * self.decay_rate_lambda * self.z + 
                          (1 - (self.alpha * self.gamma * self.decay_rate_lambda * self.prod_of_linear_matrix(self.z, x))) * x
                          )
                
                # w += \alpha * (\delta + Q - Q_old)z - \alpha * (Q - Q_old) * x
                self.w += self.alpha * (delta + q - self.old_q) * self.z - self.alpha * (q - self.old_q) * x
                
                # Q_old = Q'
                self.old_q = new_q
                
                # x = X'
                x = new_x
                
                # a = a'
                a = new_a
                
                step += 1
                with np.printoptions(precision=4, suppress=True):
                    print(new_q)
        

mc = MountainCar(n=8)
mc.create_protopoint()
# pprint.pprint(mc.protopoint)

mc.true_online_Sarsa(1000)

        
    