import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
import time
from environment import CustomTradingEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import copy
from torch.utils.data import TensorDataset, DataLoader

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(Model, self).__init__()
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        x, _ = self.lstm(x, (h0, c0))
        x = self.linear(x[:, -1, :])
        return x

class Utils():
    def __init__(self) -> None:
        pass

    def initialize_new_game(self, env, agent,no_episode):
        """We don't want an agents past game influencing its new game, so we add in some dummy data to initialize"""
        env.frame_bound = ((no_episode%70)*30+12,(no_episode%70)*30+30)
        env.reset()
        starting_frame = env.step(0)[0]

        dummy_action = 0
        dummy_reward = 0
        dummy_done = False
        for i in range(1):
            agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)


    def take_step(self,env, agent, score, debug=False):
        
        #1 and 2: Update timesteps and save weights
        agent.total_timesteps += 1
        if agent.total_timesteps % 50000 == 0:
            torch.save(agent.model.state_dict(), 'recent_weights.pt')
            print('\nWeights saved!')

        #3: Take action
        next_frame, next_frames_reward,next_deprecated, next_frame_terminal, info = env.step(agent.memory.actions[-1])
        
        #4: Get next state
        
        #5: Get next action, using next state
        next_action = agent.get_action(next_frame)
        

        #6: Now we add the next experience to memory
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
        
        #7: If game is over, return the score
        if next_frame_terminal:
            return (score + next_frames_reward),True,info

        #9: If the threshold memory is satisfied, make the agent learn from memory
        if len(agent.memory.frames) > agent.starting_mem_len:
            agent.learn(debug)

        return (score + next_frames_reward),False,info

    def play_episode(self, env, agent,no_episode, debug = False):
        self.initialize_new_game( env, agent,no_episode)
        done = False
        score = 0
        while True:
            score,done,info = self.take_step(env,agent,score, debug)
            if done:
                break
        return score,info


class Memory():
    def __init__(self,max_len):
        self.max_len = max_len
        self.frames = deque(maxlen = max_len)
        self.actions = deque(maxlen = max_len)
        self.rewards = deque(maxlen = max_len)
        self.done_flags = deque(maxlen = max_len)

    def add_experience(self,next_frame, next_frames_reward, next_action, next_frame_terminal):
        self.frames.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_frames_reward)
        self.done_flags.append(next_frame_terminal)
        
class Agent():
    def __init__(self,possible_actions,starting_mem_len,max_mem_len,starting_epsilon,learn_rate, starting_lives = 5, debug = False):
        self.memory = Memory(max_mem_len)
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9/100000
        self.epsilon_min = .05
        self.gamma = .95
        self.learn_rate = learn_rate
        self.model = self._build_model()
        self.model_target = copy.deepcopy(self.model)
        self.total_timesteps = 0
        self.lives = starting_lives #this parameter does not apply to pong
        self.starting_mem_len = starting_mem_len
        self.learns = 0
        self.model.eval()
        self.model_target.eval()

    def _build_model(self):
        model = Model(input_size=11, hidden_size=64, num_layers=2, output_size=len(self.possible_actions), dropout=0.5)
        
        print('\nAgent Initialized\n')
        return model

    def get_action(self,state):
        """Explore"""
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]

        """Do Best Acton"""
        a_index = torch.argmax(self.model(torch.tensor(np.array(state).reshape(1,12,11),dtype=torch.float32)))
        
        return self.possible_actions[a_index]

    def _index_valid(self,index):
        if self.memory.done_flags[index]:
            return False
        else:
            return True

    def fit_model(self,states,labels):
        self.model.train()
        tensor_dateset = TensorDataset(torch.tensor(np.array(states),dtype=torch.float32),torch.tensor(labels,dtype=torch.float32)) 
        dataloader = DataLoader(tensor_dateset,batch_size=32,shuffle=False)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=0.001)
        
        for epoch in range(1):
            for i, (x_batch,y_batch) in enumerate(dataloader):
                self.model.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs,y_batch)
                loss.backward()
                optimizer.step()
                
        self.model.eval()
        
    def learn(self,debug = False):
        """we want the output[a] to be R_(t+1) + Qmax_(t+1)."""
        """So target for taking action 1 should be [output[0], R_(t+1) + Qmax_(t+1), output[2]]"""

        """First we need 32 random valid indicies"""
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < 32:
            index = np.random.randint(4,len(self.memory.frames) - 1)
            if self._index_valid(index):
                state = self.memory.frames[index]
                next_state = self.memory.frames[index+1]

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index+1])
                next_done_flags.append(self.memory.done_flags[index+1])

        """Now we get the ouputs from our model, and the target model. We need this for our target in the error function"""
        states_tensor = torch.tensor(np.array(states),dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states),dtype=torch.float32)
        labels = self.model(states_tensor)
        next_state_values = self.model_target(next_states_tensor)
        
        """Now we define our labels, or what the output should have been
           We want the output[action_taken] to be R_(t+1) + Qmax_(t+1) """
        for i in range(32):
            # trebuie modificat deoarece prezic pentru fiecare exemplu 12 etichete in loc de 1
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + int((not next_done_flags[i])) * self.gamma * max(next_state_values[i])

        """Train our model using the states and outputs generated"""
        self.fit_model(states,labels)

        """Decrease epsilon and update how many times our agent has learned"""
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        
        """Every 10000 learned, copy our model weights to our target model"""
        if self.learns % 10000 == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            print('\nTarget model updated')