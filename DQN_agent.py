import torch
import random
from collections import deque, namedtuple
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging
from replay_buffer import Default_ReplayBuffer


logger=logging.getLogger("dqn_logger")
logger.setLevel(logging.WARNING)
stram_handler_formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stram_handler = logging.StreamHandler()
stram_handler.setFormatter(stram_handler_formatter)
logger.addHandler(stram_handler)

### Parameters==========================================
#Parameters definition
batch_size= 128
gamma = 0.99
tau= 0.999
lr=5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon=0.01
min_epsilon= 0.00
eps_decay_rate=1e5
print(f"device:{device}")
#=======================================================


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            # print(f"weight:{weight}, bias:{bias}")
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


Transition=namedtuple('Transition',('state','action','reward','next_state'))
class DuelingQNet(nn.Module):
    def __init__(self, n_action):
        super(DuelingQNet, self).__init__()
        self.layer1 = nn.Conv2d(4, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        self.fc = nn.Linear(64*7*7, 512)
        self.fc2 = NoisyLinear(512, 64)
        self.q = NoisyLinear(64, n_action)
        self.v = NoisyLinear(64, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.layer3)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.sum(-1, keepdim=True))
        return q
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class PrioritizedReplayBuffer:
    def __init__(self, capacity,alpha=0.6,beta=0.4):
        self.memory=deque([],maxlen=capacity)
        self.priority=deque([],maxlen=capacity)
        self.alpha=alpha
        self.beta=beta

    def add(self, priority, *args):
        self.memory.append(Transition(*args))
        self.priority.append(priority)
        return
    def update_priority(self, indices,priorities):
        for index,priority in zip(indices,priorities):
            self.priority[index]=priority
            #print(self.priority)
        return
    # TODO: Implement the sample method
    def sample(self,batch_size):
        indices=self._get_sample_indices(batch_size)
        #print(indices)
        experiences=[self.memory[i] for i in indices]
        weights=self._calculate_weights(indices).to(device)
        return experiences,indices,weights

    def _get_sample_indices(self, batch_size):
        priority_np = np.array(self.priority, dtype=np.float32)
        #logging.warning(f"Priority was: {self.priority}")
        probs = priority_np ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        return indices.tolist()

    def _calculate_weights(self, indices):
        self.beta=min(self.beta + 0.00001, 1.0)
        priorities = np.array(self.priority)[indices]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        N = len(self.priority)
        weights = (N * probs) ** (-self.beta)
        # normalize
        weights /= weights.max()
        return torch.tensor(weights,dtype=torch.float32)
    
    def __len__(self):
        return len(self.memory)

# TODO: Implement your own DQN variant here, you may also need to create other classes
class DQN_Mario:
    def __init__(self, action_size,load_path=None):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
        #self.state_size=state_size
        self.action_size=action_size
        self.q_net=DuelingQNet(action_size).to(device)
        self.target_net=DuelingQNet(action_size).to(device)
        if load_path!=None:
            self.q_net.load_state_dict(torch.load(f=load_path,weights_only=True,map_location=device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.steps_done = 0
        self.optimizer= torch.optim.Adam(self.q_net.parameters(),lr=lr,eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=lr,
                                                             total_steps= int(1e9),
                                                             pct_start= 0.01,
                                                             anneal_strategy= 'linear',
                                                             div_factor= 25,
                                                             final_div_factor= 10000,
                                                             )
        self.replaybuffer=PrioritizedReplayBuffer(100000)
        self.time_steps=0      


    def get_action(self, state, deterministic=True):
        # TODO: Implement the action selection
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).to(device)
        sample=random.random()
        eps_threshold=min_epsilon + (epsilon-min_epsilon)*np.exp(-1*self.steps_done/eps_decay_rate)
        self.steps_done+=1
        with torch.no_grad():
            action=self.q_net(state)
        if sample>=eps_threshold or deterministic:
            #print(f"debug_qnet_output:{self.q_net(state)}, {self.q_net(state).max(1)}")
            return action.argmax(dim=1, keepdim=True).cpu().item()
        else:
            return np.random.choice(range(12))


    def update(self):
        # TODO: Implement hard update or soft update
        #Hard Update
        self.target_net.load_state_dict(self.q_net.state_dict())

        #Soft Update
        # target_net_state_dict= self.target_net.state_dict()
        # policy_net_state_dict= self.q_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = tau * target_net_state_dict[key] + (1-tau) * policy_net_state_dict[key]
        # self.target_net.load_state_dict(target_net_state_dict)
        return
    def train(self):
        if len(self.replaybuffer) < 30000:
            return
        experiences, indices, weights = self.replaybuffer.sample(batch_size)
        batch = Transition(*zip(*experiences))
        state_batch = torch.tensor(np.stack(batch.state)).float().to(device)
        action_batch = torch.tensor(np.stack(batch.action)).long().to(device).unsqueeze(1)
        reward_batch = torch.tensor(np.stack(batch.reward)).float().to(device).unsqueeze(1)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.tensor(np.stack([s for s in batch.next_state if s is not None])).float().to(device)

        state_action_values= self.q_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=device)
        if non_final_mask.any():
            with torch.no_grad():
                next_action_indices= self.q_net(non_final_next_states)
                next_action_indices = next_action_indices.argmax(dim=1, keepdim=True)
                temp_next_state_values= self.target_net(non_final_next_states)
                temp_next_state_values = temp_next_state_values.gather(1, next_action_indices).squeeze(1)
                next_state_values[non_final_mask] = temp_next_state_values
        next_state_values = next_state_values.unsqueeze(1)

        expected_rewards = reward_batch + gamma  * next_state_values
        td_errors = expected_rewards - state_action_values
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()
        priority=np.squeeze(td_errors.abs().cpu().detach().numpy(), axis=-1)+0.1
        self.replaybuffer.update_priority(indices, priority.tolist())
        self.time_steps += 1
        if self.time_steps % 10000 == 0:
            self.update()
