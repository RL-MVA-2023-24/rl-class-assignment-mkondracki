from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

from collections import namedtuple, deque

import random
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import os

from evaluate import evaluate_HIV, evaluate_HIV_population
    
from itertools import count
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        #print(Transition(*args))
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    # act greedy
    def act(self, observation, use_random=False):
        device = "cuda" if next(self.policy.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            state = torch.Tensor(observation).unsqueeze(0).to(device)
            gr_action = self.policy(state).max(1).indices.view(1, 1)
            return torch.tensor([[gr_action]], device=device, dtype=torch.long)

    def save(self, path):
        self.path = path + "/model.pt"
        torch.save(self.policy.state_dict(), self.path)
        return 

    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() + "/model.pt"
        print("path : ", self.path)
        self.policy = DQN(env.observation_space.shape[0], n_actions=env.action_space.n)
        self.policy.load_state_dict(torch.load(self.path, map_location=device))
        self.policy.eval()
        return
    

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lr = 0.000001
        num_episodes_eval = 1
        self.num_episodes = 10 #10
        self.episode = 0
        self.step = 0
        eval_start = 0
        self.update_target_freq = 1000
        self.nb_opti_steps = 3

        self.batch_size = 800 #2048
        self.gamma = 0.95

        #for exploration prob computation
        self.eps_0 = 0.01
        self.eps_min = 0.01
        self.alt_eps = True

        best_model_dict = None
        self.train_rewards=[]
        self.test_rewards=[]
        self.memory = ReplayMemory(100000)

        #initializing networks
        self.policy = DQN(env.observation_space.shape[0], n_actions=env.action_space.n).to(device)
        #self.target = DQN(env.observation_space.shape[0], n_actions=env.action_space.n).to(device)
        self.target = deepcopy(self.policy).to(device)

        #########################################
        self.path = os.getcwd() + "/model3.pt"
        #self.policy = DQN(env.observation_space.shape[0], n_actions=env.action_space.n)
        self.policy.load_state_dict(torch.load(self.path, map_location=device))
        self.target = deepcopy(self.policy).to(device)

        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.multi = False

        print("num_ep : {}, num_ep_eval : {}, eval_start : {}, batch_size : {}, gamma : {}".format(self.num_episodes, num_episodes_eval, eval_start, self.batch_size, self.gamma))

        
        train_rewards = []
        test_rewards = []
        best_dict = []

        # Get the number of state observations
        state, info = env.reset()

        #for self.episode in range(self.num_episodes):
        while self.episode < self.num_episodes :
            cum_reward = self.run_episode(learn=True, batch_size=self.batch_size, device=device) #running an train episode
            train_rewards.append(cum_reward)
            print("ep : {} , cum reward : {:.2g} , eps : {:.2f}, mem : {}".format(self.episode, cum_reward, self.eps_threshold, len(self.memory)), sep='')

            #starting evaluation
            if (self.episode > eval_start):
                cum_rewards_test=[]
                cum_rewards_test = evaluate_HIV(agent=self, nb_episode=1)
                pop_test = evaluate_HIV_population(agent=self, nb_episode=1)
                #for j_episode in range(num_episodes_eval):
                #    cum_reward = self.run_episode(learn=False, batch_size=self.batch_size, device=device)
                #    cum_rewards_test.append(cum_reward)
                test_rewards.append(cum_rewards_test)

                #saving best model params if current reward better than all in the list
                if np.mean(cum_rewards_test)>=1e10:
                    best_model_dict = self.policy.state_dict()
                    print('Best model saved')  
                    self.save(os.getcwd())
                #print(f'Episodes runned : {self.episode}\tMean reward : {np.mean(cum_rewards_test):.2g}')
                print("ep : {} , cum reward : {:.2g} , eps : {:.2f}, val rew : {:.5g}, pop : {:.5g}".format(self.episode, cum_reward, self.eps_threshold, np.mean(cum_rewards_test), np.mean(pop_test)), sep='')
            #print("batch_size : ", len(self.memory))

            self.episode += 1

            

        self.policy.load_state_dict(best_model_dict) #loading the best model params into the policy net
        print('Best model loaded')    
        print('Complete')


        train_rewards.append(trainer.train_rewards)
        test_rewards.append(trainer.test_rewards)
        best_dict.append(trainer.policy.state_dict())

        self.policy.load_state_dict(best_dict[np.argmax(np.max(test_rewards, axis=1))])
        path = os.getcwd()
        self.save(path)

        print("eval : {:.2g}".format(evaluate_HIV(agent=self, nb_episode=1)))
        print("pop : {:.2g}".format(evaluate_HIV_population(agent=self, nb_episode=1)))
    


    """run a simulation that can train or eval a policy"""
    def run_episode(self, learn, batch_size, device):
        # Initialize the environment and get it's state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)   #we load the state
        cum_reward=0
        # supppose to terminate at a condition
        for t in count():
            state = torch.flatten(state).unsqueeze(0)
            action = self.select_action(state, learn, device)  #we compute an action according to policy (can be exploratory when learn=True)
            observation, reward, terminated, trunc, _ = env.step(action) #compute step, observation will be the next state
            #next_state, reward, done, trunc, _ 
            cum_reward += reward
            reward = torch.tensor([reward]).to(device)


            if terminated:
                #print("done1")
                next_state = None
            else:
                next_state = torch.flatten(torch.tensor(observation, dtype=torch.float32).to(device)).unsqueeze(0)
                #next_state = observation

            # Store the transition in memory
            if learn:
                self.memory.push(state, action, next_state, reward)
                for i in range(self.nb_opti_steps): 
                    self.optimize_step(batch_size, device)

            state = next_state

            if learn and self.step%self.update_target_freq==0:
                self.target.load_state_dict(self.policy.state_dict())
            
            self.step+=1

            if terminated or trunc:
                return cum_reward
        


    def select_action(self, state, learn, device):
        rd = random.random()
        
        if not learn:
            with torch.no_grad():
                gr_action = self.policy(state).max(1).indices.view(1, 1)
                return torch.tensor([[gr_action]], device=device, dtype=torch.long)
        
        #compute exploratory factor
        if self.alt_eps:
            #self.eps_threshold=np.max([eps_0*(self.num_episodes-self.episode)/self.num_episodes, eps_min])
            self.eps_threshold=np.max([self.eps_0*(self.num_episodes-2*self.episode)/self.num_episodes, self.eps_min])
        else:
            self.eps_threshold = self.eps_0

        if rd > self.eps_threshold:   #exploit
            with torch.no_grad():
                gr_action = self.policy(state).max(1).indices.view(1, 1)
                #print("greedy : ", torch.tensor([[gr_action]], device=device, dtype=torch.long).item())
                return torch.tensor([[gr_action]], device=device, dtype=torch.long)
        else:   #explore
            if self.multi:
                rd_action  = np.random.randint(0, 2, size=4)
            else:
                rd_action = env.action_space.sample()
                #rd_action  = np.random.randint(0, self.env.action_space.n)
            #print("explore : ", torch.tensor([[rd_action]], device=device, dtype=torch.long).item())
            return torch.tensor([[rd_action]], device=device, dtype=torch.long)
            #return torch.tensor([[np.random.randint(0,self.n_action_space)]], device=device, dtype=torch.long) 


    def optimize_step(self, batch_size, device):
        # if not enough memory quit the function
        if len(self.memory) < batch_size:
            return
        # get BATCH_SIZE transitions
        transitions = self.memory.sample(batch_size)
        #print(transitions)
        batch = Transition(*zip(*transitions))
        
        # Compute the non-final nect state : False = final next state
        #                                    True = no final
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool).to(device)
        # concatenation of all non final neext states
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state) # put all the transitions in a torch tab
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        next_state_values = torch.zeros(batch_size).to(device)
        if self.multi:
            state_action_values = self.policy(state_batch).gather(1, action_batch).sum(2)  

            with torch.no_grad():
                # selects the maximum predicted value for each next state
                next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].sum(1)
        
        else:
            state_action_values = self.policy(state_batch).gather(1, action_batch.to(torch.long))
            #print("state_action_values : ", state_action_values.shape)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        #print("expected_state_action_values : ", expected_state_action_values.unsqueeze(1).shape)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        #torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()


