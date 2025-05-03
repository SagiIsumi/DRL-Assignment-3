import torch
import numpy as np
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

size = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Default_ReplayBuffer:
    def __init__(self, 
                 storage=LazyMemmapStorage(size, device=torch.device("cpu")), 
                 sampler = PrioritizedSampler(max_capacity=size, alpha=0.6, beta=0.4),
                 batch_size=128):
        self.memory= TensorDictReplayBuffer(storage=storage, 
                                            sampler=sampler,
                                            priority_key="td_error",
                                            batch_size=batch_size)
    def add(self, priority, state, action, reward, next_state, done):
        state=torch.tensor(state, dtype=torch.float32)
        action=torch.tensor([action], dtype=torch.long)
        reward=torch.tensor([reward], dtype=torch.float32)
        next_state=torch.tensor(next_state, dtype=torch.float32)
        done=torch.tensor([done])
        priority=torch.tensor([priority], dtype=torch.float32)
        self.memory.add(TensorDict({"state": state, 
                                    "action": action, 
                                    "reward": reward,
                                    "next_state": next_state,
                                    "done": done,
                                    "td_error": priority}, 
                                    batch_size=[]))
    def sample(self, batch_size):
        return self.memory.sample(batch_size).to(device)
    def __len__(self):
        return len(self.memory)
