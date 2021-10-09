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

with open('treechop.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

# Hyperparmeters
AGENT_NUM = args['agent_num']
num_channels = args['num_channels']
ENV_NAME = args['env_name']
GAMMA = args['gamma']
LR = args['lr']

root_path = os.curdir
model_path = root_path + '/trained_model/'

writer = SummaryWriter('runs/apex/test')

@ray.remote(num_gpus=0.3)
class Testor:
    def __init__(self, model_dict, idx, num_channels=3, num_actions=19):
        import gym
        import minerl
        self.testor_idx = idx
        self.env = gym.make(ENV_NAME)
        self.port_number = int("12340") + self.testor_idx
        print("testor environment %d initialize successfully" % self.testor_idx)
        self.env.make_interactive(port=self.port_number, realtime=False)

        self.testor_network = DQN(num_channels, num_actions).cuda()
        self.testor_network.load_state_dict(model_dict)
        print("testor network %d initialize successfully" % self.testor_idx)

        self.writer = SummaryWriter(f'runs/apex/test/testor{self.testor_idx}')

        self.max_epi = 100

    def explore(self):
        for num_epi in range(self.max_epi):
            obs = self.env.reset()
            state = converter(ENV_NAME, obs).cuda()
            state = state.float()
            done = False
            total_reward = 0
            steps = 0
            total_steps = 0


            while not done:
                steps += 1
                total_steps += 1
                action_index = self.actor_network.sample_action(state, self.epsilon)
                action = make_19action(self.env, action_index)
                obs_prime, reward, done, info = self.env.step(action)
                total_reward += reward
                state_prime = converter(ENV_NAME, obs_prime).cuda()
                state = state_prime
                if done:
                    print("%d episode is done" % num_epi)
                    print("total rewards : %d " % total_reward)
                    self.writer.add_scalar('Rewards/test', total_reward, num_epi)
                    break

def test():
    ray.init()
    learner_net = DQN(num_channels=3, num_actions=19)
    model_name = model_path + 'apex_dqfd_learner2'
    model_dict = torch.load(model_name)
    testor_list = [Testor.remote(model_dict, idx, 3, 19) for idx in range(3)]
    result = [testor.remote.explore() for testor in testor_list]
    ray.get(result)
