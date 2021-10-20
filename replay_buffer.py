import random
from collections import deque
import torch
import ray
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)
        self.buffer_limit = buffer_limit

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_list = []
        action_list = []
        reward_list =[]
        next_state_list = []
        done_mask_list = []
        n_rewards_list = []

        for _, transition in enumerate(mini_batch):
            s, a, r, s_prime, done_mask, n_rewards = transition
            state_list.append(s)
            action_list.append([a])
            reward_list.append([r])
            next_state_list.append(s_prime)
            done_mask_list.append([done_mask])
            n_rewards_list.append([n_rewards])

        a = state_list
        b = torch.tensor(action_list, dtype=torch.int64)
        c = torch.tensor(reward_list)
        d = next_state_list
        e = torch.tensor(done_mask_list)
        f = torch.tensor(n_rewards_list)

        return [a, b, c, d, e, f]

    def size(self):
        return len(self.buffer)


from st import SumTree
class Memory:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        print("Memory is initialized")
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def size(self):
        return self.tree.n_entries

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / (self.tree.total() + 1e-5)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= (is_weight.max() + 1e-5)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


@ray.remote
class RemoteMemory:  # stored as ( s, a, r, s_, n_rewards ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        print("Memory is initialized")
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def size(self):
        return self.tree.n_entries

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / (self.tree.total()+ 1e-5)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= (is_weight.max() + 1e-5)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

def append_sample(memory, model, target_model, state, action, reward, next_state, done, n_rewards=None):
    # Caluclating Priority (TD Error)
    target = model(state).data.cpu()
    old_val = target[0][action].cpu()
    target_val = target_model(next_state.float()).data.cpu()
    if done:
        target[0][action] = reward
    else:
        target[0][action] = reward + 0.99 * torch.max(target_val)

    error = abs(old_val - target[0][action])
    error = error.cpu()
    if isinstance(memory, Memory):
        if n_rewards == None:
            memory.add(error, [state, action, reward, next_state, done])
        else:
            memory.add(error, (state, action, reward, next_state, done, n_rewards))

    else:
        if n_rewards == None:
            memory.remote.add(error, [state, action, reward, next_state, done])
        else:
            memory.add.remote(error, (state, action, reward, next_state, done, n_rewards))

# 여기 있으면 안된다. actor에 있거나 util에 있어야 할듯.

import sys
from typing import Dict, List, Tuple

class EpisodeMemory():
    def __init__(self, random_update=False,
                 max_epi_num=100, max_epi_len=500,
                 batch_size=1, n_step=None):
        self.random_update = random_update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.n_step = n_step

        if (random_update is False) and self.batch_size > 1:
            sys.exit('It is recommend to use 1 batch for sequential update, \
            if you want, erase this code block and modify code')

        self.memory = deque(maxlen=self.max_epi_len)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        if self.random_update:
            sampled_episodes = random.sample(self.memory, self.batch_size)
            flag = True
            min_step = self.max_epi_len

            for i, episode in enumerate(sampled_episodes):
                min_step = min(min_step, len(episode))

            for i, episode in enumerate(sampled_episodes):
                if min_step > self.n_step: #
                    idx = np.random.randint(0, len(episode)-self.n_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.n_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        else:
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])

    def __len__(self):
        return len(self.memory)

class EpisodeBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []

    def put(self, transition):
        self.obs.append(transition[0]) # 1, 4, 64, 64
        self.actions.append(transition[1])
        self.rewards.append(transition[2])
        self.next_obs.append(transition[3])
        self.dones.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None):
        obs = torch.stack(self.obs).to(device).float() # [seq, batch, 4, 64, 64]
        action = torch.stack(self.actions).to(device) # [seq, batch, 1]
        reward = torch.stack(self.rewards).to(device).float() # [seq, batch, 1]
        next_obs = torch.stack(self.next_obs).to(device).float() # [seq, batch, 1]
        done = torch.stack(self.dones).to(device) # [seq, batch, 11]

        if random_update:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs, acts=action, rews=reward, next_obs=next_obs, done=done)

    def __len__(self) -> int:
        return len(self.obs)