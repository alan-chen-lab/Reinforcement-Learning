# Problem: A robot needs to go from Start to Target
# Reward: +100 for target, -1 for each other step
# Q-value: for simplicity zero
# Initial Q values are all zero
import copy
from operator import index
from matplotlib.pyplot import grid
import numpy as np
import random
np.random.seed(9527)
random.seed(9527)

class Environment():
    def __init__(self):
        self.grid_world = [[  "T",  "s1",  "s2",  "s3",  "s4",  "s5"],
                           [ "s6",  "s7",  "s8",  "s9",   "W", "s10"],
                           ["s11",   "W", "s12",   "W", "s13", "s14"],
                           ["s15", "s16", "s17", "s18", "s19", "s20"]] #T: Target, W: Wall
        self.rows = len(self.grid_world)        #4
        self.cols = len(self.grid_world[0])     #6
        self.action_to_number = {"up": 0, "right":1, "down":2, "left":3, "fly":4}
        self.action_dict = {"up": [-1,0], "right": [0, 1], "down": [1,0], "left":[0,-1], "fly":[0,0]}
        self.direction_dict = {0: "up", 1:"right", 2:"down", 3:"left", 4:"fly"}
        self.invalid_start = ["T", "W"]
        
    def transfer_state(self, state_coordinates, action): #Input(state, action), Output(next state)
        current_state_coordinates = state_coordinates
        next_state_coordinates = state_coordinates + self.action_dict[action]
        if action=="fly" :
            if (current_state_coordinates==np.array([2,4])).all():
                return np.array([0,1])
            if (current_state_coordinates==np.array([1,3])).all():
                return np.array([1,0])
        if next_state_coordinates[0] < 0 or next_state_coordinates[0] >= self.rows or next_state_coordinates[1] < 0 or next_state_coordinates[1] >= self.cols:# Out of board
            return current_state_coordinates
        next_state = self.grid_world[next_state_coordinates[0]][next_state_coordinates[1]]
        if next_state == "W": #Hit the wall 
            return current_state_coordinates
        return next_state_coordinates

class Monte_Carlo():
    def __init__(self):
        self.env = Environment()
        self.Max_iteration = 10000
        self.gamma = 0.9
        self.alpha=0.5
        self.Horizon = 1 #Max episode_length
        self.action_len=len(self.env.direction_dict)
        self.epsilon=0.2
        self.Q_values = {}
        #self.ep=self.epsilon/6000000
        self.train=False
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.Q_values[ ((row, col), act) ] = 0 #Initialize Q value (state, action) 

        self.returns_dict = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.returns_dict[ ((row, col), act) ] = [0, 0] #[Mean value, Visited count]

    def generate_initial_state(self): #Generate randomm state
        while True:
            state_row = np.random.randint(self.env.rows)
            state_col = np.random.randint(self.env.cols)
            if self.env.grid_world[state_row][state_col] in self.env.invalid_start:
                continue
            else:
                break
        return np.array([state_row, state_col])

    def generate_random_action(self):
        action = self.env.direction_dict[np.random.randint(self.action_len)]
        while (self.env.transfer_state(self.current_state_coordinates, action) == self.current_state_coordinates).all():
                action = self.env.direction_dict[np.random.randint(self.action_len)]
        return action

    def policy(self, state_coordinate): #Optimal policy, find the maximun Q(s,a) and return action  
        Q_value = []
        valid_actions = []
        state_coordinate = np.array(state_coordinate)
        indexes = []
        
        for action in self.env.action_dict.keys():
            if not (state_coordinate == self.env.transfer_state(np.array(state_coordinate), action)).all():
                valid_actions.append(action)

        for valid_action in valid_actions:
            Q_value.append(self.Q_values[(tuple(state_coordinate), valid_action)])
        max_value = max(Q_value)
        
              
        
        if self.train:
            if np.random.rand() >= self.epsilon:
                for valid_action in valid_actions:
                    if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                        indexes.append(self.env.action_to_number[valid_action])
                return self.env.direction_dict[random.choice(indexes)]  
                # indexes = [index for index,x in enumerate(Q_value) if x == max_value]
            else :
                for valid_action in valid_actions:
                    indexes.append(self.env.action_to_number[valid_action])
                return self.env.direction_dict[random.choice(indexes)]  
        else :
            for valid_action in valid_actions:
                if max_value == self.Q_values[(tuple(state_coordinate), valid_action)]:
                    indexes.append(self.env.action_to_number[valid_action])
            return self.env.direction_dict[random.choice(indexes)]  

    def to_T(self):
        self.current_state_coordinates = self.generate_initial_state()
        self.action = self.generate_random_action()
        self.rd_state_flag=False

    def iter(self): #Main loop
        next_action=0
        self.train=True
        self.rd_state_flag=False
        self.current_state_coordinates = self.generate_initial_state()
        self.action = self.generate_random_action()
        # move iterration time
        for iterration in range(self.Max_iteration):
            #when arrive T -> random(state,action)
            if self.rd_state_flag:
                self.to_T()  
            #get next (state,action)
            next_state=self.env.transfer_state(self.current_state_coordinates, self.action)

            #collect Value function's parameter
            v=self.Q_values[(tuple(self.current_state_coordinates),self.action)]
            if (next_state==np.array([0,0])).all():
                reword=100
                next_v=0
                self.rd_state_flag=True
            else :
                reword=-1
                next_action=self.policy(next_state)
                next_v=self.Q_values[(tuple(next_state),next_action)]
                                
            #update Value
            v=v+self.alpha*(reword+self.gamma*next_v-v)
                            
            #calculate mean of Value
            returns = self.returns_dict[(tuple(self.current_state_coordinates), self.action)]
            mean = returns[0]
            visited_count = returns[1]
            mean = (mean*visited_count + v)/(visited_count + 1)
            visited_count += 1

                                    
            #save Value
            self.returns_dict[(tuple(self.current_state_coordinates), self.action)] = [mean, visited_count]
            self.Q_values[(tuple(self.current_state_coordinates), self.action)] = mean
            
            #update(state,action)                                         
            self.action=next_action
            self.current_state_coordinates=next_state

    def render(self): #Show results
        self.train=False
        output=copy.deepcopy(self.env.grid_world)
        #output = self.env.grid_world

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] in self.env.invalid_start:
                    continue
                else:
                    action = self.policy((row,col))
                    output[row][col] = action
        
        for row in range(0, self.env.rows):
            print("-------------------------------------------------------")
            out = "| "
            for col in range(0, self.env.cols):
                out += str(output[row][col]).ljust(6) + " | "
            print(out)
        print("-------------------------------------------------------")

    def demo(self): #Slides example
        self.train=True
        self.Horizon = 10
        initial_state_actions = [[[1,1], "left"], [[1,2], "right"], [[2,2], "up"], [[1,1], "up"]]
        
        for i in range(2500):
            for state_action in initial_state_actions:
                episode = []
                self.current_state_coordinates = np.array(state_action[0])
                action = state_action[1]
                for h in range(self.Horizon): #Generate episode
                    next_state_coordinates = self.env.transfer_state(self.current_state_coordinates, action)
                    reward = -1
                    if self.env.grid_world[next_state_coordinates[0]][next_state_coordinates[1]] == "T":
                        reward = 100
                        episode.append([self.current_state_coordinates, action, reward])
                        break
                    episode.append([self.current_state_coordinates, action, reward]) #Episode [[[coordinate],action, reward]]
                    self.current_state_coordinates = next_state_coordinates
                    action = self.policy(self.current_state_coordinates)
                G = 0
                
                #print("episode sequence:", episode)
                
                for h in range(len(episode)-1, -1, -1): #Iterate H-1, H-1,...,0
                    coordinate, action, reward = episode[h]
                    G = self.gamma*G + reward
                    returns = copy.deepcopy(self.returns_dict[(tuple(coordinate), action)])
                    mean = returns[0]
                    visited_count = returns[1]
                    mean = (mean*visited_count + G)/(visited_count + 1)
                    visited_count += 1
                    self.returns_dict[(tuple(episode[h][0] ), episode[h][1])] = [mean, visited_count]
                    self.Q_values[(tuple(episode[h][0]), episode[h][1])] = self.returns_dict[(tuple(episode[h][0]), episode[h][1])][0]
                '''
                print("Returns:")
                for h in range(len(episode)-1, -1, -1):
                    coordinate, action, reward = episode[h]
                    print("Coordinate:", coordinate,"\tAction:", action, "\tReturn:", self.returns_dict[(tuple(coordinate), action)])
                print("")
                '''

if __name__ == "__main__":
    print("Find optimal policy using Monte Carlo algorithm with exploring starts")
    monte = Monte_Carlo()
    print("Before (random policy), T = Target, W = Wall")
    monte.render()
    monte.iter()
    print("\nOptimal policy, T = Target, W = Wall")
    monte.render()
    print("\n\n\n")
'''
    print("************************Slides example***********************")
    monte = Monte_Carlo()
    initial_policy = [((0,1),"left"), ((0,2), "left"), ((0,3), "left"), ((0,4), "right"), ((0,5), "left"),
                      ((1,0), "right"), ((1,1), "right"), ((1,2), "down"), ((1,3), "up"), ((1,5), "up"),
                      ((2,0), "up"), ((2,2), "down"), ((2,4), "right"), ((2,5), "up"),
                      ((3,0), "up"), ((3,1), "left"), ((3,2),"left"), ((3,3), "right"), ((3,4), "left"), ((3,5), "left")]
    for state_action in initial_policy:
        monte.Q_values[state_action] = 0
    print("Before (random policy), T = Target, W = Wall")
    monte.render()
    print("\n\nRunning the episodes...\n\n")
    monte.demo()
    print("After running the episodes, T = Target, W = Wall")
    monte.render()
    '''
    # print("Q_value",monte.print_Qvalue((3,1)))






