import minerl
import gym

from model import DQN
import os
import wandb
import ray
import random
import numpy as np
from _collections import deque
from subprocess import call
from utils import *


call(["wandb", "login", "e694c5143ff8b3ba1e2b275f0ddff63443464b98"])
wandb.init(project='pre_train', entity='neverparadise')

#하이퍼 파라미터
learning_rate = 0.0003
gamma = 0.999
buffer_limit = 50000
L1 = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.curdir + '/dqn_model/'




def append_sample(memory, model, target_model, state, action, reward, next_state, done, n_rewards):
    # Caluclating Priority (TD Error)
    target = model(state.float()).data
    old_val = target[0][action].cpu()
    target_val = target_model(next_state.float()).data.cpu()
    if done:
        target[0][action] = reward
    else:
        target[0][action] = reward + 0.99 * torch.max(target_val)

    error = abs(old_val - target[0][action])
    error = error.cpu()
    memory.add(error, [state, action, reward, next_state, done, n_rewards])

ray.init()

policy_net = DQN(19).to(device=device)
target_net = DQN(19).to(device=device)
target_net.load_state_dict(policy_net.state_dict())
demos = Memory.remote(50000)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
print("pre_train start")
pre_train("MineRLTreechop-v0", demos, policy_net, target_net, optimizer, threshold=10, num_epochs=1, batch_size=128, seq_len=30, gamma=0.99, model_name='pre_trained4.pth')
print("pre_train finished")
print(ray.get(demos.size.remote()))