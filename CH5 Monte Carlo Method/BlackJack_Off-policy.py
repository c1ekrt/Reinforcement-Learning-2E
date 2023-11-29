# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:06:17 2023

@author: jim

Off-policy MC control for estimating \pi \approx \pi_*
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
    def __init__(self):
        self.board_player_ace_hit = np.zeros((10,10),dtype=float)
        self.board_player_no_ace_hit = np.zeros((10,10),dtype=float)
        self.board_player_ace_stick = np.zeros((10,10),dtype=float)
        self.board_player_no_ace_stick = np.zeros((10,10),dtype=float)
        self.board_player_ace_hit_count = np.zeros((10,10),dtype=float)
        self.board_player_no_ace_hit_count = np.zeros((10,10),dtype=float)
        self.board_player_ace_stick_countnt = np.zeros((10,10),dtype=float)
        self.board_player_no_ace_stick_count = np.zeros((10,10),dtype=float)
        self.win_reward = 1.0
        self.draw_reward = 0.0
        self.lose_reward = -1.0
        # 1 hit | stick 0
        self.hit_stick_ratio=0.4

    def find_position(self, player, dealer):
        if dealer >= 10: dealer = 9
        return player.total_point-12, dealer
    
    def find_position_int(self, player:int, dealer):
        if dealer >= 10: dealer = 9
        return player-12, dealer
    
    def behavior_policy(self):  # 1:hit 0:stick   
        return 1 if random.random() > self.hit_stick_ratio else 0
    
    def target_policy_ace_action_hit(self, pos) :
        if self.board_player_ace_hit[pos]<self.board_player_ace_stick[pos]:
            return 0
        else: return 1.0/(1.0-self.hit_stick_ratio)
    def target_policy_ace_action_stick(self, pos) :
        if self.board_player_ace_hit[pos]>self.board_player_ace_stick[pos]:
            return 0
        else: return 1.0/self.hit_stick_ratio
    def target_policy_no_ace_action_hit(self, pos) :
        if self.board_player_no_ace_hit[pos]<self.board_player_no_ace_stick[pos]:
            return 0
        else: return 1.0/(1.0-self.hit_stick_ratio)
    def target_policy_no_ace_action_stick(self, pos) :
        if self.board_player_no_ace_hit[pos]>self.board_player_no_ace_stick[pos]:
            return 0
        else: return 1.0/self.hit_stick_ratio
    
    def incremental_implement(self, old_value, new_value):
        return new_value-old_value
    
    def update(self, reward, card, player_history): # player_history: player.total_point, action, player.ace_count
        # print(player_history)
        accumulate_ratio = 1 # W
        for p in player_history:
            pos = self.find_position_int(p[0], card)
            # print(p)
            # print(pos)
            # print(accumulate_ratio)
            if p[2] > 0 : # ace count
                if p[1] == 1: # hit
                    self.board_player_ace_hit_count[pos] += accumulate_ratio
                    self.board_player_ace_hit[pos] += self.incremental_implement(accumulate_ratio * self.board_player_ace_hit[pos], reward)/self.board_player_ace_hit_count[pos]
                    if self.target_policy_ace_action_hit(pos) == 0:
                        break
                    accumulate_ratio = accumulate_ratio * self.target_policy_ace_action_hit(pos) # W = W(1/b(At,St))
            if p[2] > 0 : # ace count
                if p[1] == 0: # stick
                    
                    self.board_player_ace_stick_count[pos] += accumulate_ratio
                    self.board_player_ace_stick[pos] += self.incremental_implement(accumulate_ratio * self.board_player_ace_stick[pos], reward)/self.board_player_ace_stick_count[pos]
                    if self.target_policy_ace_action_stick(pos) == 0:
                        break
                    accumulate_ratio = accumulate_ratio * self.target_policy_ace_action_stick(pos)
            if p[2] == 0 : # ace count
                if p[1] == 1: # hit
                    self.board_player_no_ace_hit_count[pos] += accumulate_ratio
                    self.board_player_no_ace_hit[pos] += self.incremental_implement(accumulate_ratio * self.board_player_no_ace_hit[pos], reward)/self.board_player_no_ace_hit_count[pos]
                    if self.target_policy_no_ace_action_hit(pos) == 0:
                        break
                    accumulate_ratio = accumulate_ratio * self.target_policy_no_ace_action_hit(pos)
            if p[2] == 0 : # ace count
                if p[1] == 0: # stick
                    self.board_player_no_ace_stick_count[pos] += accumulate_ratio
                    self.board_player_no_ace_stick[pos] += self.incremental_implement(accumulate_ratio * self.board_player_no_ace_stick[pos], reward)/self.board_player_no_ace_stick_count[pos]
                    if self.target_policy_no_ace_action_stick(pos) == 0:
                        break
                    accumulate_ratio = accumulate_ratio * self.target_policy_no_ace_action_stick(pos)
        pass
        
    def game_start(self):
        player_history = [] # every-time visit method
        
        player = BlackJack()
        dealer = BlackJack()
        player.hit()
        player.hit()
        card = dealer.hit()
        
        while player.total_point <= 11:
            player.hit()
        # print(player.total_point," ", end='')
        while player.total_point <= 21:
            action = self.behavior_policy() # 1:hit 0:stick
            player_history.append((player.total_point, action, player.ace_count)) # tuple
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
            return self.lose_reward, card, player_history
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
        
    def run_game(self, episode=1):
        for e in range(0, episode):
            play.game_start()
    
            
        
play = Game()
play.run_game(10000000)
    
 
print(play.board_player_no_ace_hit)
print(play.board_player_ace_hit)
print(play.board_player_no_ace_stick)
print(play.board_player_ace_stick)
print(play.board_player_no_ace_hit_count)
print(play.board_player_ace_hit_count)
print(play.board_player_no_ace_stick_count)
print(play.board_player_ace_stick_count)
    