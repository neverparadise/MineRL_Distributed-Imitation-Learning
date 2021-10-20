import minerl
import gym
from model import DRQN
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from replay_buffer import EpisodeBuffer, EpisodeMemory
from utils import *
import torch

def train(policy_net, target_net, memory, optimizer, batch_size, gamma):
    samples, seq_len = memory.sample() # samples : episode transition list, (transition : dict)
    observations = [] # each obs : [seq, 1, 4, 64, 64]
    actions = [] # each actions : [seq, 1]
    rewards = [] # each rewards : [seq, 1]
    next_observations = [] # each nobs : [seq, 1, 4, 64, 64]
    dones = [] # each dones : [seq, 1]

    for i in range(batch_size):
        observations.append(samples[i]["obs"]) #
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]['rews'])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = torch.cat(observations, dim=1).to(device) # concat with batch dim [seq, batch, 4, 64, 64]
    actions = torch.cat(actions, dim=1).to(device)
    rewards = torch.cat(rewards, dim=1).to(device)
    next_observations = torch.cat(next_observations, dim=1).to(device)
    dones = torch.cat(dones, dim=1).to(device)

    target_hidden = target_net.init_hidden_state(False, batch_size=batch_size,
                                          training=True)

    total_loss = torch.tensor(0.0).to(device)
    for i in range(seq_len):
        next_obs = next_observations[i]
        q_target, target_hidden = target_net(next_obs, target_hidden.to(device))
        q_target_max = q_target.max(2)[0].detach()
        targets = rewards[i] + gamma*q_target_max*dones[i]

        obs = observations[i]
        hidden = policy_net.init_hidden_state(batch_first=False,
                                              batch_size=batch_size,
                                              training=True)
        q_out, hidden = policy_net(obs, hidden.to(device))
        temp_action = actions[i].clone()
        temp_action = temp_action.unsqueeze(0) # 1, 2, 1
        q_a = q_out.gather(2, temp_action) # q_out : (1, 2, 6) tensor. actions : (1, 2)
        loss = F.smooth_l1_loss(q_a, targets)
        total_loss += loss
    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()
    print("gradient is updated")

def run():
    model_name = "drqn_pomdp_random"
    env_name = "MineRLNavigateDense-v0"
    seed = 1

    env = gym.make(env_name)
    env.make_interactive(realtime=False, port=6666)
    device = torch.device("cuda")
    np.random.seed(seed)
    random.seed(seed)
    writer = SummaryWriter('runs/'+env_name+"_"+model_name)

    batch_size = 2
    learning_rate = 1e-3
    memory_size = 50000
    min_epi_num = 1
    target_update_period = 2

    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2

    random_update = True
    n_step = 10
    max_epi = 10000
    max_epi_len = 10000
    max_epi_step = 30000

    num_channels = 4
    batch_first = False
    policy_net = DRQN(num_channels=4, num_actions=6, batch_first=batch_first).cuda().float()
    target_net = DRQN(num_channels=4, num_actions=6, batch_first=batch_first).cuda().float()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    score = 0
    total_score = 0

    epsilon = eps_start

    memory = EpisodeMemory(random_update=random_update,
                           max_epi_num=100, max_epi_len=max_epi_len,
                           batch_size=batch_size,
                           n_step=n_step)


    for e in range(max_epi):
        state = env.reset()
        obs = converter(env_name, state).to(device) # obs : [1, 4, 64, 64]
        done = False

        episode_record = EpisodeBuffer()
        hidden = policy_net.init_hidden_state(batch_first=batch_first,
                                              batch_size=batch_size,
                                              training=False)
        for t in range(max_epi_step):
            action_index, hidden = policy_net.sample_action(obs, epsilon, hidden)
            action = make_6action(env, action_index)
            s_prime, reward, done, info = env.step(action)
            obs_prime = converter(env_name, s_prime).to(device)
            done_mask = 0.0 if done else 1.0

            batch_action = torch.tensor([action_index]).unsqueeze(0).to(device)
            batch_reward = torch.tensor([reward]).unsqueeze(0).to(device)
            batch_done = torch.tensor([done_mask]).unsqueeze(0).to(device)
            episode_record.put([obs, batch_action, batch_reward/10.0,
                                obs_prime, batch_done])
            obs = obs_prime
            score += reward
            total_score += reward

            if len(memory) > min_epi_num:
                print(len(memory))
                train(policy_net,target_net,memory,optimizer,batch_size, gamma=0.99)

                if (t + 1) % target_update_period == 0:
                    for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):  # <- soft update
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

            if done:
                print(f"Score of # {e} episode : {score}")
                break
        memory.put(episode_record)
        epsilon = max(eps_end, epsilon * eps_decay)

        if e % 5:
            torch.save(policy_net, model_name + '.pth')
        writer.add_scalar('Rewards per episodes', score, e)
        score = 0

    writer.close()
    env.close()

run()