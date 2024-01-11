from collections import deque
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, bidirectional = False):
        super(Model, self).__init__()
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        # Multiply hidden_size by 2 because of bidirectional
        self.linear = nn.Linear(hidden_size, output_size)
        
        # Xavier/Glorot initialization for linear layer
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        batch_size = x.size(0)
        # Add these lines to handle reshaping
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        
        # multiply self.num_layers by 2 if bidirectional
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = F.relu(x)  # Add ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.linear(x[:, -1, :])
        return x

    def predict(self, x):
        """
        Predict the action for the given state.

        Parameters:
        - x: Input state

        Returns:
        - action: Predicted action
        - q_values: Q-values for all possible actions
        """
        # Ensure the model is in evaluation mode
        self.eval()

        # Preprocess the input state if needed (e.g., unsqueeze if necessary)
        if len(x.size()) == 2:
            x = x.unsqueeze(1)

        # Pass the input through the model
        with torch.no_grad():
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            output, _ = self.lstm(x, (h0, c0))
            output = F.relu(output)
            output = self.dropout(output)
            q_values = self.linear(output[:, -1, :])

        # Choose the action with the highest Q-value
        action = torch.argmax(q_values).item()

        return action, q_values
    
        
class Utils:
    def __init__(self) -> None:
        pass

    def initialize_new_game(self, env, agent, no_episode):
        """
        Initialize a new game environment by resetting it and adding dummy data to the agent's memory.

        Parameters:
        - env: Gym trading environment
        - agent: Agent object
        - no_episode: Episode number
        """
        env.reset()
        starting_frame = env.step(0)[0]

        # Add dummy data to agent's memory
        dummy_action = 0
        dummy_reward = 0
        dummy_done = False
        for i in range(1):
            agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

    def take_step(self, env, agent, score, debug=False):
        """
        Take a step in the environment, update agent's memory, and perform learning if conditions are met.

        Parameters:
        - env: Gym trading environment
        - agent: Agent object
        - score: Current episode score
        - debug: Debug mode flag

        Returns:
        - Tuple containing the updated score, a flag indicating if the episode is done, and environment info.
        """
        # Update timesteps and save weights periodically
        agent.total_timesteps += 1
        if agent.total_timesteps % 50000 == 0:
            torch.save(agent.model.state_dict(), 'more_recent_weights.pt')
            print('\nWeights saved!')

        # Take action in the environment
        next_frame, next_frames_reward, _, next_frame_terminal, info = env.step(agent.memory.actions[-1])

        # Get next action using the agent's policy
        next_action = agent.get_action(next_frame)

        # Add the next experience to agent's memory
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

        # If the game is over, return the final score
        if next_frame_terminal:
            return (score + next_frames_reward), True, info

        # If the memory threshold is satisfied, make the agent learn
        if len(agent.memory.frames) > agent.starting_mem_len:
            agent.learn(debug)

        return (score + next_frames_reward), False, info

    def play_episode(self, env, agent, no_episode, debug=False):
        """
        Play a full episode in the environment.

        Parameters:
        - env: Gym trading environment
        - agent: Agent object
        - no_episode: Episode number
        - debug: Debug mode flag

        Returns:
        - Tuple containing the final episode score, total profit, and environment info.
        """
        self.initialize_new_game(env, agent, no_episode)
        done = False
        score = 0
        while True:
            score, done, info = self.take_step(env, agent, score, debug)
            if done:
                break
        return score, info


class Memory:
    def __init__(self, max_len):
        """
        Initialize the memory replay buffer.

        Parameters:
        - max_len: Maximum length of the replay buffer
        """
        self.max_len = max_len
        self.frames = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.done_flags = deque(maxlen=max_len)

    def add_experience(self, next_frame, next_frames_reward, next_action, next_frame_terminal):
        """
        Add a new experience to the replay buffer.

        Parameters:
        - next_frame: Next state/frame of the environment
        - next_frames_reward: Reward obtained in the next time step
        - next_action: Action taken in the current time step
        - next_frame_terminal: Flag indicating if the episode terminated after the next frame
        """
        self.frames.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_frames_reward)
        self.done_flags.append(next_frame_terminal)

class Agent():
    def __init__(self, possible_actions, starting_mem_len, max_mem_len, starting_epsilon, learn_rate, device, starting_lives=5, debug=False):
        """
        Initialize the DQN agent.

        Parameters:
        - possible_actions: List of possible actions the agent can take
        - starting_mem_len: Initial length of the experience replay buffer
        - max_mem_len: Maximum length of the experience replay buffer
        - starting_epsilon: Initial exploration rate
        - learn_rate: Learning rate for the neural network
        - starting_lives: Initial number of lives (if applicable, default is 5)
        - debug: Flag for debugging information (default is False)
        """
        self.memory = Memory(max_mem_len)
        self.device = device
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = 0.9 / 100000
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.learn_rate = learn_rate
        self.model = self._build_model().to(self.device)
        self.model_target = copy.deepcopy(self.model).to(self.device)
        self.total_timesteps = 0
        self.lives = starting_lives  # This parameter may not apply to all environments
        self.starting_mem_len = starting_mem_len
        self.learns = 0
        self.model.eval()
        self.model_target.eval()

    def _build_model(self):
        """
        Build the neural network model.

        Returns:
        - model: The neural network model
        """
        model = Model(input_size=10, hidden_size=64, num_layers=2, output_size=len(self.possible_actions), dropout=0.2)
        
        print('\nAgent Initialized\n')
        return model

    def get_action(self, state):
        """
        Choose an action based on the current state.

        Parameters:
        - state: Current state of the environment

        Returns:
        - action: Chosen action
        """
        # Explore
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]

        # Exploit (choose the best action)
        state_tensor = torch.tensor(np.array(state).reshape(1, 30, 10), dtype=torch.float32).to(self.device) 
        action_index = torch.argmax(self.model(state_tensor))
        return self.possible_actions[action_index]

    def _index_valid(self, index):
        """
        Check if the given index is valid in the replay buffer.

        Parameters:
        - index: Index to be checked

        Returns:
        - valid: True if the index is valid, False otherwise
        """
        return not self.memory.done_flags[index]
    
    def fit_model(self, states, labels):
        """
        Train the neural network model using the provided states and labels.

        Parameters:
        - states: Input states for training
        - labels: Target labels for training
        """
        self.model.train()
        tensor_dataset = TensorDataset(torch.tensor(np.array(states), dtype=torch.float32).to(self.device) ,
                                    torch.tensor(labels, dtype=torch.float32).to(self.device))
        dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

        for epoch in range(1):
            for i, (x_batch, y_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()

    def learn(self, debug=False):
        """
        Update the agent's knowledge and adjust its behavior based on experiences in the replay memory.

        Parameters:
        - debug: Flag for debugging information (default is False)
        """
        # We want the output[a] to be R_(t+1) + Qmax_(t+1).
        # So the target for taking action 1 should be [output[0], R_(t+1) + Qmax_(t+1), output[2]]

        # V1
        # First, we need 32 random valid indice
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < 32:
            index = np.random.randint(4, len(self.memory.frames) - 1)
            if self._index_valid(index):
                state = self.memory.frames[index]
                next_state = self.memory.frames[index + 1]

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index + 1])
                next_done_flags.append(self.memory.done_flags[index + 1])    
        
        # Now we get the outputs from our model and the target model.
        # We need this for our target in the error function
        # V1-2        
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        labels = self.model(states_tensor)
        next_state_values = self.model_target(next_states_tensor)
        
        # V1-2
        # Now we define our labels, or what the output should have been.
        # We want the output[action_taken] to be R_(t+1) + Qmax_(t+1)
        for i in range(32):
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + int((not next_done_flags[i])) * self.gamma * max(next_state_values[i])

        # Train our model using the states and outputs generated
        self.fit_model(states, labels)

        # Decrease epsilon and update how many times our agent has learned
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1

        # Every 10000 learned, copy our model weights to our target model
        if self.learns % 10000 == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            print('\nTarget model updated')
