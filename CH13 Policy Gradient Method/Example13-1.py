import random
import numpy as np
import matplotlib.pyplot as plt

class ShortCorridor:
    def __init__(self):
        self.action = 2
        self.state_direction = [[0,1],[1,-1],[-1,1]]
        self.right_prob_start = 0.05
        self.right_prob_end = 0.96
    
    def run(self):
        state = 0
        prob = []
        steps = []
        right_cur_prob = self.right_prob_start
        ttr = 25000
        while self.right_prob_end > right_cur_prob:
            step = 0
            for i in range(0, ttr):
                
                while state != 3:
                    step += -1
                    rd = random.random()
                    if rd > right_cur_prob:
                        state += self.state_direction[state][0]
                    else:
                        state += self.state_direction[state][1]
                state = 0
            prob.append(round(right_cur_prob,2))
            steps.append(step/float(ttr))
            right_cur_prob += 0.01
            
        prob = np.array(prob)
        steps = np.array(steps)
        ymax = steps.max()
        xmax = prob[np.argmax(steps)]
        
        plt.xlim(0,1)
        plt.ylim(-90,0)
        plt.plot(prob,steps,'r')
        text = "x={:.2f}, y={:.2f}".format(xmax, ymax)

        plt.annotate(text,  xy=(xmax, ymax), xytext=(0.45,-20), 
                     arrowprops={
                     'width':1,
                     'headlength':8,
                     'headwidth':10,
                     'facecolor':'#000',
                     'shrink':0.05,
                     },
                     fontsize=10
                     )
        plt.show()
        
sc = ShortCorridor()
sc.run()