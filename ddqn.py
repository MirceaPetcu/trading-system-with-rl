import numpy as np
import gymnasium as gym
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input,LSTM, Dropout
from tensorflow.keras.optimizers import AdamW
import keras.backend as K
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import time
from environment import CustomTradingEnv

class Utils():
    def __init__(self) -> None:
        pass

    def initialize_new_game(self, env, agent):
        """We don't want an agents past game influencing its new game, so we add in some dummy data to initialize"""
        
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
          agent.model.save_weights('recent_weights.hdf5')
          print('\nWeights saved!')

        #3: Take action
        next_frame, next_frames_reward,next_deprecated, next_frame_terminal, info = env.step(agent.memory.actions[-1])
        
        #4: Get next state
        # nu mai trebuie sa pun frame-urile trecute manual deoarece TradingEnv le ia oricum una dupa alta de window size 
        # (se repeta oricum--> sample efficient)
        new_state = []
        new_state.append(next_frame)
        
        #5: Get next action, using next state
        next_action = agent.get_action(next_frame)
        

        #6: Now we add the next experience to memory
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
        
        #7: If game is over, return the score
        if next_frame_terminal:
            return (score + next_frames_reward),True

        #8: If we are trying to debug this then render
        if debug:
            # print(new_state)
            # print(next_frames_reward)
            # print(next_frame_terminal)
            pass

        #9: If the threshold memory is satisfied, make the agent learn from memory
        if len(agent.memory.frames) > agent.starting_mem_len:
            agent.learn(debug)

        return (score + next_frames_reward),False

    def play_episode(self, env, agent, debug = False):
        self.initialize_new_game( env, agent)
        done = False
        score = 0
        while True:
            score,done = self.take_step(env,agent,score, debug)
            if done:
                break
        return score


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
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.lives = starting_lives #this parameter does not apply to pong
        self.starting_mem_len = starting_mem_len
        self.learns = 0


    def _build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape = (12,11),activation='relu', return_sequences = True))
        model.add(Dropout(0.2)) 
        model.add(LSTM(64, return_sequences = False,activation='relu'))
        model.add(Dense(64,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(len(self.possible_actions), activation = 'linear'))
        optimizer = AdamW(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialized\n')
        return model

    def get_action(self,state):
        """Explore"""
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]

        """Do Best Acton"""
        state = np.reshape(state,(1,12,11))
        a_index = np.argmax(self.model.predict(state,verbose = 0))
        return self.possible_actions[a_index]

    def _index_valid(self,index):
        if self.memory.done_flags[index]:
            return False
        else:
            return True

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
        labels = self.model.predict(np.array(states),verbose = 0)
        next_state_values = self.model_target.predict(np.array(next_states),verbose=0)
        
        """Now we define our labels, or what the output should have been
           We want the output[action_taken] to be R_(t+1) + Qmax_(t+1) """
        for i in range(32):
            # trebuie modificat deoarece prezic pentru fiecare exemplu 12 etichete in loc de 1
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + int((not next_done_flags[i])) * self.gamma * max(next_state_values[i])

        """Train our model using the states and outputs generated"""
        # print('train')
        self.model.fit(np.array(states),labels,batch_size = 32, epochs = 1, verbose = 0)

        """Decrease epsilon and update how many times our agent has learned"""
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        
        """Every 10000 learned, copy our model weights to our target model"""
        if self.learns % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')