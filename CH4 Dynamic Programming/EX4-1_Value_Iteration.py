import numpy as np

class State:
    def __init__(self, sizex:int, sizey:int, terminal_block, iteration:int, reward:float):
        '''
        sizex:(positive int)
            size of x
        sizey:(positive int)
            size of y
        terminal block:(2d list of positive int in (n,2) shape)
            the block that end the episode when stepped on
        iteration:(positive int)
            how many times it loops to get closer to v_\pi
        reward:(real number)
            reward gain from every step.
        '''
        self.value = np.zeros((sizex,sizey),dtype=np.float32)           
        self.ex_value = np.zeros((sizex,sizey),dtype=np.float32)
        self.sizex = sizex
        self.sizey = sizey
        for t in terminal_block:
            if t[0]>=sizex or t[1]>=sizey or t[0]<0 or t[1]<0:
                ValueError
            else:
                self.value[t[0]][t[1]] = 1 
        self.terminal_block = terminal_block
        self.iteration = iteration
        self.reward = reward
        #valid move include: up,down,left,right
        self.valid_move = np.array([[1,0],[0,1],[-1,0],[0,-1]])     
        
        
    def add_block_in_direction(self, which_row_or_col:int, direction:str):
        '''
        which_row_or_column:(int that doesn't exceed sizex or sizey depend on row or column you want to add)
            row or column you would like to add
        direction:(valid string between["up", "down", "left", "right"])
            direction of row or column you would like to add
            
        '''
        if direction=="up":
            new_row = np.full((1, self.sizey),2.0)
            new_row[0][which_row_or_col] = 0
            new_map = np.append(new_row,self.value,axis=0)
            self.sizex += 1
        elif direction=="down":
            new_row = np.full((1, self.sizey),2.0)
            new_row[0][which_row_or_col] = 0
            new_map = np.append(self.value,new_row,axis=0)
            self.sizex += 1
        elif direction=="left":
            new_row = np.full((self.sizex, 1),2.0)
            new_row[which_row_or_col][0] = 0
            new_map = np.append(new_row,self.value,axis=1)
            self.sizey += 1
        elif direction=="right":
            new_row = np.full((self.sizex, 1),2.0)
            new_row[which_row_or_col][0] = 0
            new_map = np.append(self.value,new_row,axis=1)
            self.sizey += 1
        else:
            ValueError
        self.value = np.array(new_map)
        self.ex_value = np.array(self.value)
        print(np.array(self.ex_value))
        
    def add_block_in_coordinate(self, target:list):
        '''
        target:(int with two element [x,y])
            add a block in coordinate (x,y), if x,y exceed map, additional map will be drawn.
        '''
        new_map = np.full((max(self.sizex,target[0]+1),max(self.sizey,target[1]+1)),2.0)
        new_map[:self.sizex, :self.sizey] = np.array(self.value)
        new_map[target[0],target[1]] = 0
        self.value = np.array(new_map)
        self.ex_value = np.array(self.value)
        print(np.array(self.ex_value))
    
    def calculate_value(self, target):########################################Value Iteration##################################################
        '''
        target:(int with two element [x,y])
            calculate the new v(s) depend on the vicinity with bellman equation.
        '''
        new_value = []
        for move in self.valid_move:
            new_target = target + move 
            if new_target[0]>=self.sizex or new_target[1]>=self.sizey or new_target[0]<0 or new_target[1]<0:
                new_target -= move
            elif self.ex_value[new_target[0]][new_target[1]] == 2:
                new_target -= move
            elif self.ex_value[new_target[0]][new_target[1]] == 1:
                return self.reward
            if self.ex_value[new_target[0]][new_target[1]] <= 0:
                new_value.append(self.ex_value[new_target[0]][new_target[1]])
        return max(new_value)+self.reward
            
    def print_value(self):
        for row in range (self.value.shape[0]):
            output = "|"
            output_line = ""
            for col in range(self.value.shape[1]):
                if self.value[row][col] == 1: 
                    output += "{:.6f}".format(0.0)
                elif self.value[row][col] == 2:
                    output += "         "
                else:
                    output += "{:.6f}".format(self.value[row][col])
                output_line += "-----------"
                output +="|"
            print(output)
            print(output_line)
        
    def iterate(self):
        for i in range (0, self.iteration):
            for row in range (self.value.shape[0]):
                for col in range(self.value.shape[1]):
                    if self.value[row][col] > 0: 
                        continue
                    new_value = self.calculate_value([row,col])
                    self.value[row][col] = new_value
            self.ex_value = np.array(self.value)
    

x = State(4, 4, [[0,0],[3,3]],5, -1)
x.iterate()
x.print_value()
