import random
from collections import deque
import torch
import ray
import numpy as np


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
