import gym
import numpy as np
import torch
from DQN_agent import DQN_Mario
import cv2

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.observation_buffer=[]
        self.timer=0
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_path="qnet_3000.pth"
        self.agent= DQN_Mario(12,load_path)
        self.buffer = np.zeros((2,240,256,3), dtype=np.uint8)
        shape=(84,84)
        self.shape=tuple(shape)
        self.action = 0
    def act(self, observation):
        #print(observation)
        self.timer+=1
        if self.timer%4==3:
            self.buffer[0]=observation
        if self.timer%4==0:
            self.buffer[1]=observation
            observation = self.buffer.max(axis=0)
        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # print(observation.shape,len(observation))
        if len(self.observation_buffer) < 4:
            self.observation_buffer.append(observation) 
        if self.timer%4==0:
            self.observation_buffer.pop(0)
            self.observation_buffer.append(observation)
            state=np.stack(self.observation_buffer, axis=0)
            self.action=self.agent.get_action(state,True)                
        return self.action