# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:17:55 2023

@author: jim
"""

import numpy as np
import random 
import time
import math
def draw_track_1():
    track_1 = np.zeros((19,34),dtype=int) 
    '''
    0 : racetrack
    1 : wall
    8 : start
    9 : end
    '''
    for y in range(1,20):
        track_1[1][y] = 1
    for y in range (29,33):
        track_1[1][y] = 1
    for y in range(1,10):
        track_1[1][y] = 1
    for y in range (30,33):
        track_1[1][y] = 1
    for y in range(1,3):
        track_1[2][y] = 1
    for y in range (32,33):
        track_1[2][y] = 1
    for x in range (10,18):   
        for y in range(1,27):
            track_1[x][y] = 1
    track_1[10][26] = 0
    # for x in range (4,10): # we do this seperately
    #     track_1[x][1]=8
    # for y in range (27,33):
    #     track_1[17][y]=9
    for y in range (0,33):
        track_1[0][y]=1
        track_1[18][y]=1
    for x in range (0,19):
        track_1[x][0]=1
        track_1[x][33]=1
    return track_1

class Racetrack:
    def __init__(self, track:int):
        self.select_track = 0
        _start_location = []
        for x in range (4,10):
            _start_location.append([x,1])
        
        _end_location = []
        for y in range (27,33):
            _end_location.append([17,y])
        
        self.start_location = np.array(_start_location)
        self.number_of_start_location = len(self.start_location)
        self.end_location = np.array(_end_location)
        self.track_1 = draw_track_1()
'''
    1 0 -1 x
  1 0 1 2
  0 3 4 5 
 -1 6 7 8
 y
0 0 : {0, 1, 3}
0 1 : {0, 1, 3, 4, 6}
0 2 : {0, 1, 3, 4, 6, 7}
0 3 : {0, 1, 3, 4, 6, 7}
0 4 : {3, 4, 6, 7}
1 0 : {0, 1, 2, 3, 4}
1 1 : {0, 1, 2, 3, 4, 5, 6, 7}
1 2 : {0, 1, 2, 3, 4, 5, 6, 7}
1 3 : {0, 1, 2, 3, 4, 5, 6, 7}
1 4 : {3, 4, 5, 6, 7}
2 0 : {0, 1, 2, 3, 4, 5}
2 1 : {0, 1, 2, 3, 4, 5, 6, 7}
2 2 : {0, 1, 2, 3, 4, 5, 6, 7}
2 3 : {0, 1, 2, 3, 4, 5, 6, 7}
2 4 : {3, 4, 5, 6, 7}
3 0 : {0, 1, 2, 3, 4, 5}
3 1 : {0, 1, 2, 3, 4, 5, 6, 7}
3 2 : {0, 1, 2, 3, 4, 5, 6, 7}
3 3 : {0, 1, 2, 3, 4, 5, 6, 7}
3 4 : {3, 4, 5, 6, 7}
4 0 : {1, 2, 4, 5}
4 1 : {1, 2, 4, 5, 7}
4 2 : {1, 2, 4, 5, 7}
4 3 : {1, 2, 4, 5, 7}
4 4 : {4, 5, 7}
 etc...
'''
def draw_action_map(max_speed=4):
    action_map = []
    valid_act = set()
    for i in range (0,8):
        valid_act.add(i)
    for x in range (0,max_speed+1):
        x_action_map = []
        for y in range(0,max_speed+1):
            cur_act = valid_act.copy()
            if x == 0 and y == 0:
                cur_act.discard(4)
            if x == 0 and y == 1:
                cur_act.discard(7)
            if x == 1 and y == 0:
                cur_act.discard(5)
            if x == 0 :
                cur_act.discard(2)
                cur_act.discard(5)
                cur_act.discard(8)
            if y == 0 :
                cur_act.discard(6)
                cur_act.discard(7)
                cur_act.discard(8)
            if x == 4 :
                cur_act.discard(0)
                cur_act.discard(3)
                cur_act.discard(6)
            if y == 4 :
                cur_act.discard(0)
                cur_act.discard(1)
                cur_act.discard(2)
            x_action_map.append(list(cur_act))
            print(f"{x} {y} : {cur_act}")
        action_map.append(x_action_map)
    return action_map

            

class Car(Racetrack):
    def __init__(self, track):
        super().__init__(track)                                                     
        self.max_speed = 4                                                                                                       
        self.speed_x = 0                                                            
        self.speed_y = 0   
        self.valid_action = draw_action_map(4)                                                         
        self.action_graph = [[1,1],[0,1],[-1,1],[1,0],[0,0],[-1,0],[1,-1],[0,-1],[-1,-1]]
        self.location = np.array([5,5])
        self.from_random_start()
        for location in self.end_location:
            self.track_1[location[0],location[1]] = 9
        # np.savetxt("test.txt",self.track_1 ,fmt='%d')
    
    def from_random_start(cls):
        cls.speed_x = 0
        cls.speed_y = 0
        start = random.randint(0, cls.number_of_start_location-1)
        cls.location = np.array(cls.start_location[start])
    
    def action(self):
        action_pool = self.valid_action[self.speed_x][self.speed_y]
        get_action = action_pool[random.randint(0,len(action_pool))-1]
        old_speed_x = self.speed_x
        old_speed_y = self.speed_y
        old_location = np.array(self.location)
        if self.check_collison():
            self.speed_x += self.action_graph[get_action][0]
            self.speed_y += self.action_graph[get_action][1]
            self.location[0] += self.speed_x
            self.location[1] += self.speed_y
        else:
            self.from_random_start()
        return (old_location[0], old_location[1], old_speed_x, old_speed_y, get_action)
        
    def check_collison(self):
        '''
        0Y011 True
        00111
        00111
        00100
        10X00
        
        0Y011 False
        00111
        01111
        00100
        10X00
        
        soft method
        left turn:(ceil method)
        -> right turn:(floor method)
        '''
        if self.location[0]+self.speed_x >= self.track_1.shape[0] or self.location[1]+self.speed_y >= self.track_1.shape[1]:
            return False
            
        loc = np.array(self.location.copy(), dtype=float)
        
        m = np.array([self.speed_x/4.0 , self.speed_y/4.0],dtype=float)
        for i in range (0,4):
            loc += m
            if self.track_1[math.floor(loc[0])][math.floor(loc[1])]==1:
                return False
        return True      
    
        
        
class Game:
    def __init__(self, track:int):
        # state include location, speed_x, speed_y, action
        # shape :       (19,34,5,5,9)
        self.v = np.zeros((19,34,5,5,9),dtype=float) # (0,0) is not considered during moving
        self.c = np.zeros((19,34,5,5,9),dtype=float) # cumulative reward
        
    def start(self):
        t = 0 # reward is negative time
        
        track = Racetrack(0)
        car = Car(track)
        history = []
        print(car.start_location)
        while True:
            cur_state_action = car.action()
            history.append(cur_state_action)
            # print(cur_state_action)   
            # time.sleep(0.5)
            if car.track_1[car.location[0]][car.location[1]]==9:
                 break
        print(history)   
        
    def update(self, history):
        pass
    
    def policy(self):
        pass
    
        
        
        
    
# track = draw_track_1()
# np.savetxt("test.txt",track ,fmt='%d')

game = Game(0)
game.start()
