import minerl
import gym
import argparse
import torch.serialization
import torch
import torch.optim as optim
import ray
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import Memory, RemoteMemory, ReplayBuffer

from model import DQN, DRQN
from utils import *

with open('navigate.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

    # Hyperparmeters
    AGENT_NUM = args['agent_num']
    num_channels = args['num_channels']
    ENV_NAME = args['env_name']
    GAMMA = args['gamma']
    LR = args['lr']

    root_path = os.curdir
    model_path = root_path + '/trained_model/'

writer = SummaryWriter('runs/dqn/')

def update_network(policy_net, target_net, memory, batch_size, optimizer, total_steps):
    batch, idxs, is_weights = memory.sample(batch_size)
    for i, transition in enumerate(batch):
        print(transition)
        s, a, r, s_prime, done_mask = transition
        s = s.float().to(device)
        a = torch.tensor([a], dtype=torch.int64).to(device)
        r = torch.tensor([r]).to(device)
        s_prime = s_prime.float().to(device)
        done_mask = torch.tensor(done_mask).float().to(device)

        q_vals = policy_net(s)
        a = a.unsqueeze(0)
        state_action_values = q_vals.gather(1, a)

        # comparing the q values to the values expected using the next states and reward
        next_state_values = target_net(s_prime).max(1)[0].unsqueeze(1)
        target = r + (next_state_values * GAMMA) * done_mask

        # calculating the q loss, n-step return lossm supervised_loss
        is_weight = torch.FloatTensor(is_weights).to(device)
        q_loss = (is_weight[i].detach() * F.l1_loss(state_action_values, target)).mean()

        errors = torch.abs(state_action_values - target).data.cpu()
        errors = errors.numpy()
        # update priority
        idx = idxs[i]
        memory.update(idx, errors)
        # optimization step and logging
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', q_loss, total_steps)

def update_target(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), model_path + "dqn.pth")

def run():
    policy_net = DQN(num_channels, 19).cuda()
    target_net = DQN(num_channels, 19).cuda()
    optimizer =  optim.Adam(policy_net.parameters(), LR)
    memory = Memory(50000)
    env = gym.make(ENV_NAME)
    env.make_interactive(port=6666, realtime=False)
    max_epi = 100
    n_step = 2
    update_period = 10
    gamma = 0.99

    total_steps = 0
    epsilon = 0.95
    endEpsilon = 0.01
    stepDrop = (epsilon - endEpsilon) / max_epi

    for num_epi in range(max_epi):
        obs = env.reset()
        state = converter(ENV_NAME, obs).cuda()
        state = state.float()
        done = False
        total_reward = 0
        steps = 0
        if epsilon > endEpsilon:
            epsilon -= stepDrop

        while not done:
            steps += 1
            total_steps += 1
            a_out = policy_net.sample_action(state, epsilon)
            action_index = a_out
            action = make_19action(env, action_index)
            obs_prime, reward, done, info = env.step(action)

            total_reward += reward


            if done:
                print("%d episode is done" % num_epi)
                print("total rewards : %d " % total_reward)
                writer.add_scalar('Rewards/train', total_reward, num_epi)
                break

            state_prime = converter(ENV_NAME, obs_prime).cuda()
            append_sample(memory, policy_net, target_net, state, action_index,
                          reward, state_prime, done)
            state = state_prime

            if memory.size() > 1000:
                update_network(policy_net, target_net, memory, 2, optimizer, total_steps)

            if total_steps % 2000 == 0:
                update_target(policy_net, target_net)



run()