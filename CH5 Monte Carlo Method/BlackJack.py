# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:06:17 2023

@author: jim
"""
import random
import numpy as np

class BlackJack:
    global point 
    point = [11,2,3,4,5,6,7,8,9,10,10,10,10]
    def __init__(self):
        self.ace_count = 0
        self.total_point = 0
    def hit(self):
        card = random.randint(0,12)
        if card == 0:
            self.ace_count += 1
            self.total_point += 11
        else: self.total_point += point[card]
        if self.ace_count>0 and self.total_point>21:
            self.ace_count -= 1
            self.total_point -= 10
        return card


class Game:
    def __init__(self, policy:int):
        self.board_player_ace_hit = np.zeros((10,10),dtype=float)
        self.board_player_no_ace_hit = np.zeros((10,10),dtype=float)
        self.board_player_ace_stick = np.zeros((10,10),dtype=float)
        self.board_player_no_ace_stick = np.zeros((10,10),dtype=float)
        self.board_player_ace_hit_count = np.zeros((10,10),dtype=int)
        self.board_player_no_ace_hit_count = np.zeros((10,10),dtype=int)
        self.board_player_ace_stick_count = np.zeros((10,10),dtype=int)
        self.board_player_no_ace_stick_count = np.zeros((10,10),dtype=int)
        self.policy_list = [self.deterministic_policy, self.soft_deterministic_policy]
        self.epsilon = 0.3
        self.policy = policy
        self.win_reward = 1.0
        self.draw_reward = 0.0
        self.lose_reward = -1.0
    
    def find_position(self, player, dealer):
        if dealer >= 10: dealer = 9
        return player.total_point-12, dealer
    
    def find_position_int(self, player:int, dealer):
        if dealer >= 10: dealer = 9
        return player-12, dealer
    
    def deterministic_policy(self, player, dealer):  # 1:hit 0:stick
        # print(self.board_player_ace_hit)
        pos = self.find_position(player, dealer)
        if player.ace_count > 0:
            if self.board_player_ace_hit[pos] == self.board_player_ace_stick[pos]: 
                return 0 if random.randint(0, 1) == 0 else 1 
            else: 
                return 0 if self.board_player_ace_hit[pos] < self.board_player_ace_stick[pos] else 1
        if player.ace_count == 0:
            if self.board_player_no_ace_hit[pos] == self.board_player_no_ace_stick[pos]: 
                return 0 if random.randint(0, 1) == 0 else 1 
            else: 
                return 0 if self.board_player_no_ace_hit[pos] < self.board_player_no_ace_stick[pos] else 1
    def soft_deterministic_policy(self, player, dealer):  # 1:hit 0:stick
        pos = self.find_position(player, dealer)
        prob = random.random()
        if player.ace_count > 0:
            if self.board_player_ace_hit[pos] == self.board_player_ace_stick[pos] or prob < self.epsilon: 
                return 0 if random.randint(0, 1) == 0 else 1 
            else: 
                return 0 if self.board_player_ace_hit[pos] < self.board_player_ace_stick[pos] else 1
        if player.ace_count == 0:
            if self.board_player_no_ace_hit[pos] == self.board_player_no_ace_stick[pos] or prob < self.epsilon: 
                return 0 if random.randint(0, 1) == 0 else 1 
            else: 
                return 0 if self.board_player_no_ace_hit[pos] < self.board_player_no_ace_stick[pos] else 1
    
    def incremental_implement(self, old_value, new_value):
        return new_value-old_value
    
    def update(self, reward, card, player_history): # player_history: player.total_point, action, player.ace_count
        # print(reward)
        for p in player_history:
            pos = self.find_position_int(p[0], card)
            # print(p,end="")
        
            # print(pos)
            if p[2] > 0 : # ace count
                if p[1] == 1: # hit
                    self.board_player_ace_hit_count[pos] += 1
                    self.board_player_ace_hit[pos] += self.incremental_implement(self.board_player_ace_hit[pos], reward)/self.board_player_ace_hit_count[pos]
            if p[2] > 0 : # ace count
                if p[1] == 0: # stick
                    self.board_player_ace_stick_count[pos] += 1
                    self.board_player_ace_stick[pos] += self.incremental_implement(self.board_player_ace_stick[pos], reward)/self.board_player_ace_stick_count[pos]
            if p[2] == 0 : # ace count
                if p[1] == 1: # hit
                    self.board_player_no_ace_hit_count[pos] += 1
                    self.board_player_no_ace_hit[pos] += self.incremental_implement(self.board_player_no_ace_hit[pos], reward)/self.board_player_no_ace_hit_count[pos]
            if p[2] == 0 : # ace count
                if p[1] == 0: # stick
                    self.board_player_no_ace_stick_count[pos] += 1
                    self.board_player_no_ace_stick[pos] += self.incremental_implement(self.board_player_no_ace_stick[pos], reward)/self.board_player_no_ace_stick_count[pos]
        pass
        
    def game_start(self):
        player_history = set() #
        
        player = BlackJack()
        dealer = BlackJack()
        player.hit()
        player.hit()
        card = dealer.hit()
        
        while player.total_point <= 11:
            player.hit()
        # print(player.total_point," ", end='')
        while player.total_point <= 21:
            action = self.policy_list[self.policy](player, card) # 1:hit 0:stick
            player_history.add((player.total_point, action, player.ace_count)) # tuple
            if action == 0:
                break
            else: 
                player.hit()
        #         print(player.total_point," ", end='')
        # print(player.total_point," |", end='')
        # print(dealer.total_point," ", end='')
        if player.total_point > 21:
            # print("L")
            self.update(self.lose_reward, card, player_history)
            return
        while dealer.total_point <= 16:
            dealer.hit()
        # print(dealer.total_point)
        if dealer.total_point > 21:
            self.update(self.win_reward, card, player_history)
            # print("W")
            return
        if dealer.total_point > player.total_point:
            # print("L")
            self.update(self.lose_reward, card, player_history)
            return
        elif dealer.total_point == player.total_point:
            # print("D")
            self.update(self.draw_reward, card, player_history)
            return
        else:
            # print("W")
            self.update(self.win_reward, card, player_history)
            return
        
    
    
    def iteration(self, count:int):
        pass
            
        
play = Game(policy=1)
for e in range (0,10000000):
    play.game_start()
 
print(play.board_player_no_ace_hit)
print(play.board_player_ace_hit)
print(play.board_player_no_ace_stick)
print(play.board_player_ace_stick)
print(play.board_player_no_ace_hit_count)
print(play.board_player_ace_hit_count)
print(play.board_player_no_ace_stick_count)
print(play.board_player_ace_stick_count)
    