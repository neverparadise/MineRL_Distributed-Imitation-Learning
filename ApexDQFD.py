from model import DQN
import os
import minerl
import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import ray

from _collections import deque
from utils import *
import random

from subprocess import call

#하이퍼 파라미터
learning_rate = 0.0003
gamma = 0.999
buffer_limit = 50000
L1 = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def margin_loss(q_value, action, demo, weigths):
    ae = F.one_hot(action, num_classes=19)
    zero_indices = (ae == 0)
    one_indices = (ae == 1)
    ae[zero_indices] = 1
    ae[one_indices] = 0
    ae = ae.to(float)
    max_value = torch.max(q_value + ae, axis=1)

    ae = F.one_hot(action, num_classes=19)
    ae = ae.to(float)

    J_e = torch.abs(torch.sum(q_value * ae,axis=1) - max_value.values)
    J_e = torch.mean(J_e * weigths * demo)
    return J_e

total_episodes = 1000
startEpsilon = 1.0
endEpsilon = 0.05
epsilon = startEpsilon

root_path = os.curdir
model_path = root_path + '/trained_model/'

stepDrop = (startEpsilon - endEpsilon) / total_episodes


@ray.remote
class Actor:
    def __init__(self, learner, actor_idx, epsilon):
        # environment initialization
        import gym
        import minerl
        self.actor_idx = actor_idx
        self.env = gym.make("MineRLTreechop-v0")
        self.port_number = int("12340") + actor_idx
        print("actor environment %d initialize successfully" % self.actor_idx)
        self.shared_network_cpu = ray.get(learner.get_network.remote())
        # self.shared_memory = ray.get(shared_memory_id)
        # print("shared memory assign successfully")

        # network initalization
        self.actor_network = DQN(19).cpu()
        self.actor_target_network = DQN(19).cpu()
        self.actor_network.load_state_dict(self.shared_network_cpu.state_dict())
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        print("actor network %d initialize successfully" % self.actor_idx)

        self.initialized = False
        self.epi_counter = 0
        # exploring info
        self.epsilon = epsilon
        self.max_step = 100
        self.local_buffer_size = 100
        self.local_buffer = deque(maxlen=self.local_buffer_size)

        project_name = 'apex_dqfd_Actor%d' %(actor_idx)
        wandb.init(project=project_name, entity='neverparadise')

    # 1. 네트워크 파라미터 복사
    # 2. 환경 탐험 (초기화, 행동)
    # 3. 로컬버퍼에 저장
    # 4. priority 계산
    # 5. 글로벌 버퍼에 저장
    # 6. 주기적으로 네트워크 업데이트

    def get_initialized(self):
        return self.initialized

    def get_counter(self):
        return self.epi_counter

    # 각 환경 인스턴스에서 각 엡실론에 따라 탐험을 진행한다.
    # 탐험 과정에서 local buffer에 transition들을 저장한다.
    # local buffer의 개수가 특정 개수 이상이면 global buffer에 추가해준다.

    def explore(self, learner, shared_memory):
        self.env.make_interactive(port=self.port_number, realtime=False)
        self.initialized = True

        for num_epi in range(self.max_step):
            obs = self.env.reset()
            state = converter(obs).cpu()
            state = state.float()
            done = False
            total_reward = 0
            steps = 0
            total_steps = 0
            self.epsilon = 0.5
            if (self.epsilon > endEpsilon):
                self.epsilon -= stepDrop / (self.actor_idx + 1)

            n_step = 2
            n_step_state_buffer = deque(maxlen=n_step)
            n_step_action_buffer = deque(maxlen=n_step)
            n_step_reward_buffer = deque(maxlen=n_step)
            n_step_n_rewards_buffer = deque(maxlen=n_step)
            n_step_next_state_buffer = deque(maxlen=n_step)
            n_step_done_buffer = deque(maxlen=n_step)
            gamma_list = [0.99 ** i for i in range(n_step)]

            while not done:
                steps += 1
                total_steps += 1
                a_out = self.actor_network.sample_action(state, self.epsilon)
                action_index = a_out
                action = make_action(self.env, action_index)
                #action['attack'] = 1
                obs_prime, reward, done, info = self.env.step(action)
                total_reward += reward
                state_prime = converter(obs_prime)

                # local buffer add
                n_step_state_buffer.append(state)
                n_step_action_buffer.append(action_index)
                n_step_reward_buffer.append(reward)
                n_step_next_state_buffer.append(state_prime)
                n_step_done_buffer.append(done)
                n_rewards = sum([gamma * reward for gamma, reward in zip(gamma_list, n_step_reward_buffer)])
                n_step_n_rewards_buffer.append(n_rewards)

                if (len(n_step_state_buffer) >= n_step):
                    # LocalBuffer Get
                    # Compute Priorities
                    for i in range(n_step):
                        self.append_sample(shared_memory, self.actor_network, self.actor_target_network, \
                                           n_step_state_buffer[i], \
                                           n_step_action_buffer[i], n_step_reward_buffer[i], \
                                           n_step_next_state_buffer[i], \
                                           n_step_done_buffer[i], \
                                           n_step_n_rewards_buffer[i])
                        if (n_step_done_buffer[i]):
                            break
                state = state_prime.float().cpu()
                if done:
                    break

            if done:
                print("%d episode is done" % num_epi)
                print("total rewards : %d " % total_reward)
                wandb.log({"rewards": total_reward})
                self.update_params(learner)

            #if (num_epi % 5 == 0 and num_epi != 0):
            #    print("actor network is updated ")

    def env_close(self):
        self.env.close()

    def update_params(self, learner):
        shared_network = ray.get(learner.get_network.remote())
        self.actor_network.load_state_dict(shared_network.state_dict())

    def append_sample(self, memory, model, target_model, state, action, reward, next_state, done, n_rewards):
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
        memory.add.remote(error, [state, action, reward, next_state, done, n_rewards])

@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, network, batch_size):
        self.learner_network = DQN(19).cuda().float()
        self.learner_target_network = DQN(19).cuda().float()
        self.learner_network.load_state_dict(network.state_dict())
        self.learner_target_network.load_state_dict(network.state_dict())
        self.shared_network = DQN(19).cpu()
        self.count = 0
        self.batch_size = batch_size
        wandb.init(project='apex_dqfd_Learner', entity='neverparadise')

    # 1. sampling
    # 2. calculate gradient
    # 3. weight update
    # 4. compute priorities
    # 5. priorities of buffer update
    # 6. remove old memory
    def count(self):
        return self.count
    def get_network(self):
        self.shared_network.load_state_dict(self.learner_network.state_dict())
        return self.shared_network

    def update_network(self, memory, demos, batch_size, optimizer, actor):
        while(ray.get(actor.get_counter.remote()) < 100):
            print("update_network")
            agent_batch, agent_idxs, agent_weights = ray.get(memory.sample.remote(batch_size))
            demo_batch, demo_idxs, demo_weights = ray.get(demos.sample.remote(batch_size))

            # demo_batch = (batch_size, state, action, reward, next_state, done, n_rewards)
            # print(len(demo_batch[0])) # 0번째 배치이므로 0이 나옴
            state_list = []
            action_list = []
            reward_list = []
            next_state_list = []
            done_mask_list = []
            n_rewards_list = []

            for agent_transition in agent_batch:
                s, a, r, s_prime, done_mask, n_rewards = agent_transition
                state_list.append(s)
                action_list.append([a])
                reward_list.append([r])
                next_state_list.append(s_prime)
                done_mask_list.append([done_mask])
                n_rewards_list.append([n_rewards])

            for expert_transition in demo_batch:
                s, a, r, s_prime, done_mask, n_rewards = expert_transition
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
            target = r + (next_state_values * gamma * done_mask)

            # calculating the q loss, n-step return lossm supervised_loss
            is_weights = torch.FloatTensor(agent_weights).to(device)
            q_loss = (is_weights * F.mse_loss(state_action_values, target)).mean()
            n_step_loss = (state_action_values.max(1)[0] + nr).mean()
            supervised_loss = margin_loss(q_vals, a, 1, 1)

            loss = q_loss + supervised_loss + n_step_loss
            errors = torch.abs(state_action_values - target).data.cpu().detach()
            errors = errors.numpy()
            # update priority
            for i in range(batch_size):
                idx = agent_idxs[i]
                memory.update.remote(idx, errors[i])

            # optimization step and logging
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.learner_network.parameters(), 100)
            optimizer.step()
            torch.save(self.learner_network.state_dict(), model_path + "apex_dqfd_learner.pth")
            self.count +=1
            if(self.count % 20 == 0 and self.count != 0):
                self.update_target_networks()
            print("leaner_network updated")
            return loss

    def update_target_networks(self):
        self.learner_target_network.load_state_dict(self.learner_network.state_dict())
        print("leaner_target_network updated")


ray.init()


policy_net = DQN(19).cuda()
target_net = DQN(19).cuda()
target_net.load_state_dict(policy_net.state_dict())
memory = Memory.remote(50000)
demos = Memory.remote(25000)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)

# Copy network params from pretrained Agent
model_path = './dqn_model/pre_trained6.pth'
policy_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
target_net.load_state_dict(policy_net.state_dict())

#parse_demo2.remote("MineRLTreechop-v0", demos, policy_net.cpu(), target_net.cpu(), optimizer, threshold=60, num_epochs=1, batch_size=4, seq_len=60, gamma=0.99, model_name='pre_trained4.pth')

# learner network initialzation
batch_size = 256
demo_prob = 0.5
learner = Learner.remote(policy_net, batch_size)

# actor network, environments initialization
# Generating each own instances
num_actors = 2
epsilon = 0.5
actor_list = [Actor.remote(learner, i, 0.5) for i in range(num_actors)]

#memory_id = ray.put(memory)
#demos_id = ray.put(demos)

explore = [actor.explore.remote(learner, memory) for actor in actor_list]

#explore = [actor.explore.remote(learner, ray.get(memory_id)) for actor in actor_list]
update = learner.update_network.remote(memory, demos, batch_size, optimizer, actor_list[0])
ray.get(explore)
ray.get(update)
