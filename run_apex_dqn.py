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

writer = SummaryWriter('runs/a3c/')


@ray.remote
class ParameterServer:
    def __init__(self, num_channels, num_actions):
        self.policy_params = DQN(num_channels, num_actions).state_dict()
        self.target_params = DQN(num_channels, num_actions).state_dict()

    def pull_from_learner(self, learner):
        learner.push_parameters.remote(self.policy_params, self.target_params)

    def push_to_actor(self):
        return self.policy_params, self.target_params

@ray.remote(num_gpus=0.2)
class Actor:
    def __init__(self, learner, param_server, actor_idx, epsilon, num_channels=3, num_actions=19, ):
        # environment initialization
        import gym
        import minerl
        self.actor_idx = actor_idx
        self.env = gym.make(ENV_NAME)
        self.port_number = int("12340") + actor_idx
        print("actor environment %d initialize successfully" % self.actor_idx)
        self.env.make_interactive(port=self.port_number, realtime=False)
        self.learner_state_dict = ray.get(learner.get_state_dict.remote())
        print("getting learner state dict finished...")
        # network initalization
        self.actor_network = DQN(num_channels, num_actions).cuda()
        self.actor_target_network = DQN(num_channels, num_actions).cuda()
        self.actor_network.load_state_dict(self.learner_state_dict)
        self.actor_target_network.load_state_dict(self.learner_state_dict)
        print("actor network %d initialize successfully" % self.actor_idx)

        self.param_server = param_server
        self.epi_counter = 0
        self.max_epi = 1000
        self.n_step = 4
        self.update_period = 10
        self.gamma = 0.99

        # exploring info
        self.epsilon = epsilon
        self.endEpsilon = 0.01
        self.stepDrop = (self.epsilon - self.endEpsilon) / self.max_epi
        self.local_buffer_size = 100
        self.local_buffer = deque(maxlen=self.local_buffer_size)

        self.writer = SummaryWriter(f'runs/apex/actor{self.actor_idx}')

        # 1. 네트워크 파라미터 복사
        # 2. 환경 탐험 (초기화, 행동)
        # 3. 로컬버퍼에 저장
        # 4. priority 계산
        # 5. 글로벌 버퍼에 저장
        # 6. 주기적으로 네트워크 업데이트


    def get_epi_counter(self):
        return self.epi_counter

    def update_params(self, learner):
        ray.get(self.param_server.pull_from_learner.remote(learner))
        policy_params, target_params = ray.get(self.param_server.push_to_actor.remote())
        self.actor_network.load_state_dict(policy_params)
        self.actor_target_network.load_state_dict(target_params)

    def append_sample(self, memory, state, action, reward, next_state, done, n_rewards=None):
        # Caluclating Priority (TD Error)
        target = self.actor_network(state).data
        old_val = target[0][action].cpu()
        target_val = self.actor_target_network(next_state).data.cpu()
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + 0.99 * torch.max(target_val)

        error = abs(old_val - target[0][action])
        error = error.cpu()
        state_ = state.cpu()
        next_state_ = next_state.cpu()

        if isinstance(memory, Memory):
            if n_rewards == None:
                memory.add(error, [state_, action, reward, next_state_, done])
            else:
                memory.add(error, (state_, action, reward, next_state_, done, n_rewards))

        else:
            if n_rewards == None:
                memory.remote.add(error, [state_, action, reward, next_state_, done])
            else:
                memory.add.remote(error, (state_, action, reward, next_state_, done, n_rewards))

    def explore(self, learner, memory):
        for num_epi in range(self.max_epi):
            obs = self.env.reset()
            state = converter(ENV_NAME, obs).cuda()
            state = state.float()
            done = False
            total_reward = 0
            steps = 0
            total_steps = 0
            if (self.epsilon > self.endEpsilon):
                self.epsilon -= self.stepDrop

            # initialize local_buffer
            n_step = self.n_step
            n_step_state_buffer = deque(maxlen=n_step)
            n_step_action_buffer = deque(maxlen=n_step)
            n_step_reward_buffer = deque(maxlen=n_step)
            n_step_n_rewards_buffer = deque(maxlen=n_step)
            n_step_next_state_buffer = deque(maxlen=n_step)
            n_step_done_buffer = deque(maxlen=n_step)
            gamma_list = [self.gamma ** i for i in range(n_step)]

            while not done:
                steps += 1
                total_steps += 1
                a_out = self.actor_network.sample_action(state, self.epsilon)
                action_index = a_out
                action = make_19action(self.env, action_index)
                obs_prime, reward, done, info = self.env.step(action)
                total_reward += reward
                state_prime = converter(ENV_NAME, obs_prime).cuda()

                # put transition in local buffer
                n_step_state_buffer.append(state)
                n_step_action_buffer.append(action_index)
                n_step_reward_buffer.append(reward)
                n_step_next_state_buffer.append(state_prime)
                n_step_done_buffer.append(done)
                n_rewards = sum([gamma * reward for gamma, reward in zip(gamma_list, n_step_reward_buffer)])
                n_step_n_rewards_buffer.append(n_rewards)

                if (len(n_step_state_buffer) >= n_step):
                    # Compute Priorities
                    for i in range(n_step):
                        self.append_sample(memory,
                                           n_step_state_buffer[i],
                                           n_step_action_buffer[i],
                                           n_step_reward_buffer[i],
                                           n_step_next_state_buffer[i],
                                           n_step_done_buffer[i],
                                           n_step_n_rewards_buffer[i])
                        if (n_step_done_buffer[i]):
                            break
                state = state_prime
                self.actor_network.cuda()
                self.actor_target_network.cuda()

                if done:
                    print("%d episode is done" % num_epi)
                    print("total rewards : %d " % total_reward)
                    self.writer.add_scalar('Rewards/train', total_reward, num_epi)
                    self.epi_counter += 1
                    if (num_epi % self.update_period == 0):
                        self.update_params(learner)
                    break


@ray.remote(num_gpus=0.4)
class Learner:
    def __init__(self, param_server, batch_size, num_channels, num_actions):
        self.learner_network = DQN(num_channels, num_actions).cuda().float()
        self.learner_target_network = DQN(num_channels, num_actions).cuda().float()
        self.count = 0
        self.batch_size = batch_size
        self.writer = SummaryWriter(f'runs/apex/learner')

        self.lr = LR
        self.optimizer = optim.Adam(self.learner_network.parameters(), self.lr)
        self.param_server = param_server

    def learning_count(self):
        return self.count

    def get_state_dict(self):
        return self.learner_network.state_dict()

    def push_parameters(self, temp_network_dict, temp_target_dict):
        temp_network_dict = (self.learner_network.state_dict())
        temp_target_dict = (self.learner_target_network.state_dict())

    def load(self, dir):
        network = torch.load(dir)
        self.learner_network.load_state_dict(network.state_dict())
        self.learner_target_network.load_state_dict(network.state_dict())

    def update_network(self, memory):

        if isinstance(memory, Memory):
            agent_batch, agent_idxs, agent_weights = memory.sample(self.batch_size)
        else:
            agent_batch, agent_idxs, agent_weights = ray.get(memory.sample.remote(self.batch_size))

        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_mask_list = []
        n_rewards_list = []
        for i, agent_transition in enumerate(agent_batch):
            s, a, r, s_prime, done_mask, n_rewards = agent_transition
            state_list.append(s)
            action_list.append([a])
            reward_list.append([r])
            next_state_list.append(s_prime)
            done_mask_list.append([done_mask])
            n_rewards_list.append([n_rewards])

        s = torch.stack(state_list).float().cuda()
        a = torch.tensor(action_list, dtype=torch.int64).cuda()
        r = torch.tensor(reward_list).cuda()
        s_prime = torch.stack(next_state_list).float().cuda()
        done_mask = torch.tensor(done_mask_list).float().cuda()
        nr = torch.tensor(n_rewards_list).float().cuda()

        q_vals = self.learner_network(s)
        state_action_values = q_vals.gather(1, a)

        # comparing the q values to the values expected using the next states and reward
        next_state_values = self.learner_target_network(s_prime).max(1)[0].unsqueeze(1)
        target = r + (next_state_values * self.gamma * done_mask)

        # calculating the q loss, n-step return lossm supervised_loss
        is_weights = torch.FloatTensor(agent_weights).to(device)
        q_loss = (is_weights * F.mse_loss(state_action_values, target)).mean()
        n_step_loss = (state_action_values.max(1)[0] + nr).mean()

        loss = q_loss + n_step_loss
        errors = torch.abs(state_action_values - target).data.cpu().detach()
        errors = errors.numpy()
        # update priority
        for i in range(self.batch_size):
            idx = agent_idxs[i]
            if isinstance(memory, RemoteMemory):
                memory.update.remote(idx, errors[i])
            else:
                memory.update(idx, errors[i])

        # optimization step and logging
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.save(self.learner_network.state_dict(), model_path + "apex_dqfd_learner.pth")
        self.count += 1
        if (self.count % 50 == 0 and self.count != 0):
            self.update_target_networks()
        print("leaner_network updated")
        self.writer.add_scalar('Loss/train', loss)
        return loss

    def update_target_networks(self):
        self.learner_target_network.load_state_dict(self.learner_network.state_dict())
        print("leaner_target_network updated")

def main():
    ray.init()
    policy_net = DQN(num_channels=4, num_actions=19)
    target_net = DQN(num_channels=4, num_actions=19)
    target_net.load_state_dict(policy_net.state_dict())
    #memory = Memory(50000)
    #shared_memory = ray.get(ray.put(memory))
    memory = RemoteMemory.remote(30000)
    num_channels = 4
    num_actions = 19
    batch_size = 256
    param_server = ParameterServer.remote(num_channels, num_actions)
    learner = (Learner.remote(param_server, batch_size, num_channels, num_actions))
    print(learner)
    print(learner.get_state_dict.remote())

    num_actors = 2
    epsilon = 0.9

    actor_list = [Actor.remote(learner, param_server, i, epsilon, num_channels, num_actions) for i in range(num_actors)]
    explore = [actor.explore.remote(learner, memory) for actor in actor_list]
    ray.get(explore)
    # while (actor.episode < 100)
    # print(f"learning count : {self.count}")
    while ray.get(learner.learning_count.remote()) < 10000000:
        if ray.get(learner.learning_count.remote()) % 100 == 0:
            print(ray.get(learner.learning_count.remote()))
        memory_size = ray.get(memory.size.remote())
        if memory_size > 2000:
            learn = learner.update_network.remote(memory)
            ray.get(learn)

main()