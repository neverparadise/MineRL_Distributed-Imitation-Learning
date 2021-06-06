import torch
import random
from _collections import deque
import gym
import minerl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import ray
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
gamma = 0.99
buffer_limit = 50000
L1 = 0.9
model_path = os.curdir + '/dqn_model/'

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

        for transition in mini_batch:
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
@ray.remote
class Memory:  # stored as ( s, a, r, s_, n_rewards ) in SumTree
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

        sampling_probabilities = priorities / (self.tree.total()+ 1e-5)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= (is_weight.max() + 1e-5)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

def train_dqn(policy_net, target_net, demos, batch_size, optimizer):
    demo_batch, idxs, is_weights = ray.get(demos.sample.remote(batch_size))
    state_list = []
    action_list = []
    reward_list = []
    next_state_list = []
    done_mask_list = []
    n_rewards_list = []

    for transition in demo_batch:
        s, a, r, s_prime, done_mask, n_rewards = transition
        state_list.append(s)
        action_list.append([a])
        reward_list.append([r])
        next_state_list.append(s_prime)
        done_mask_list.append([done_mask])
        n_rewards_list.append([n_rewards])

    s = torch.stack(state_list).float().to(device)
    a = torch.tensor(action_list, dtype=torch.int64).to(device)
    r = torch.tensor(reward_list).to(device)
    s_prime = torch.stack(next_state_list).float().to(device)
    done_mask = torch.tensor(done_mask_list).float().to(device)
    nr = torch.tensor(n_rewards_list).to(device)

    q_vals = policy_net(s)
    state_action_values = q_vals.gather(1, a)

    # comparing the q values to the values expected using the next states and reward
    next_state_values = target_net(s_prime).max(1)[0].unsqueeze(1)
    target = r + (next_state_values * gamma) * done_mask

    # calculating the q loss, n-step return lossm supervised_loss
    is_weights = torch.FloatTensor(is_weights).to(device)
    q_loss = (is_weights * F.mse_loss(state_action_values, target)).mean()
    n_step_loss = (state_action_values.max(1)[0] + nr).mean()
    supervised_loss = margin_loss(q_vals, a, 1, 1)

    loss = q_loss + supervised_loss + n_step_loss
    wandb.log({"Q-loss": q_loss.item()})
    wandb.log({"n-step loss": n_step_loss.item()})
    wandb.log({"super_vised loss": supervised_loss.item()})
    wandb.log({"total loss": loss.item()})

    errors = torch.abs(state_action_values - target).data.cpu()
    errors = errors.numpy()
    # update priority
    for i in range(batch_size):
        idx = idxs[i]
        demos.update.remote(idx, errors[i])

    # optimization step and logging
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 100)
    optimizer.step()
    return loss

def make_action(env, action_index):
    # Action들을 정의
    action = env.action_space.noop()
    if (action_index == 0):
        action['camera'] = [0, -5]
        action['attack'] = 0
    elif (action_index == 1):
        action['camera'] = [0, -5]
        action['attack'] = 1
    elif (action_index == 2):
        action['camera'] = [0, 5]
        action['attack'] = 0
    elif (action_index == 3):
        action['camera'] = [0, 5]
        action['attack'] = 1
    elif (action_index == 4):
        action['camera'] = [-5, 0]
        action['attack'] = 0
    elif (action_index == 5):
        action['camera'] = [-5, 0]
        action['attack'] = 1
    elif (action_index == 6):
        action['camera'] = [5, 0]
        action['attack'] = 0
    elif (action_index == 7):
        action['camera'] = [5, 0]
        action['attack'] = 1

    elif (action_index == 8):
        action['forward'] = 0
        action['jump'] = 1
    elif (action_index == 9):
        action['forward'] = 1
        action['jump'] = 1
    elif (action_index == 10):
        action['forward'] = 1
        action['attack'] = 0
    elif (action_index == 11):
        action['forward'] = 1
        action['attack'] = 1
    elif (action_index == 12):
        action['back'] = 1
        action['attack'] = 0
    elif (action_index == 13):
        action['back'] = 1
        action['attack'] = 1
    elif (action_index == 14):
        action['left'] = 1
        action['attack'] = 0
    elif (action_index == 15):
        action['left'] = 1
        action['attack'] = 1
    elif (action_index == 16):
        action['right'] = 1
        action['attack'] = 0
    elif (action_index == 17):
        action['right'] = 1
        action['attack'] = 1
    else:
        action['attack'] = 1

    return action

def converter(observation):
        obs = observation['pov']
        obs = obs / 255.0
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        return obs

def converter2(observation):
    obs = observation / 255.0
    obs = torch.from_numpy(obs)
    obs = obs.permute(2, 0, 1)
    return obs

def parse_demo(env_name, rep_buffer, threshold=10, nsteps=10, num_epochs=1, batch_size=16, seq_len=10, gamma=0.99):
    data = minerl.data.make(env_name)
    demo_num = 0
    for s_batch, a_batch, r_batch, ns_batch, d_batch in data.batch_iter(num_epochs=num_epochs, batch_size=batch_size,
                                                                        seq_len=seq_len):
        demo_num += 1
        if r_batch.sum() < threshold:
            continue

        """
        state_batch : (batch_size, seq_len, 64, 64, 3)
        action_batch : (batch_size, seq_len, action['property'].shape) ex camera = 2 otherwise 1

        reward_batch : (batch_size, seq_len)
        next_state_batch : (batch_size, seq_len, 64, 64, 3)
        done_batch : (batch_size, seq_len)


        reward, _ = stats.mode(r_batch, axis=1)
        reward = np.squeeze(reward)
        done, _ = stats.mode(d_batch, axis=1)
        done = np.squeeze(done)
        """
        parse_ts = 0

        # 각 state에 대한 action discretize를 위해 반복문을 사용
        batch_length = (s_batch['pov'].shape)[0]  # (batch, seq, 64, 64, 3)[0]
        for i in range(0, batch_length):
            episode_start_ts = 0

            n_step = seq_len
            n_step_state_buffer = deque(maxlen=n_step)
            n_step_action_buffer = deque(maxlen=n_step)
            n_step_reward_buffer = deque(maxlen=n_step)
            n_step_n_rewards_buffer = deque(maxlen=n_step)
            n_step_next_state_buffer = deque(maxlen=n_step)
            n_step_done_buffer = deque(maxlen=n_step)
            gamma_list = [gamma ** i for i in range(n_step)]

            for j in range(0, seq_len):
                av = a_batch['attack'][i][j]  # attack value
                aj = a_batch['jump'][i][j]  # jump value
                af = a_batch['forward'][i][j]  # forward value
                ab = a_batch['back'][i][j]  # back value
                al = a_batch['left'][i][j]  # left value
                ar = a_batch['right'][i][j]  # right value
                va = a_batch['camera'][i][j][0]  # vertical angle and
                ha = a_batch['camera'][i][j][1]  # horizontal angle

                camera_thresholds = (abs(va) + abs(ha)) / 2.0
                # 카메라를 움직이는 경우
                if (camera_thresholds > 2.5):
                    # camera = [0, -5]
                    if abs(va) < abs(ha) and ha < 0:
                        if av == 0:
                            action_index = 0
                        else:
                            action_index = 1
                    # camera = [0, 5]
                    elif abs(va) < abs(ha) and ha > 0:
                        if av == 0:
                            action_index = 2
                        else:
                            action_index = 3
                    # camera = [-5, 0]
                    elif abs(va) > abs(ha) and ha < 0:
                        if av == 0:
                            action_index = 4
                        else:
                            action_index = 5
                    # camera = [5, 0]
                    elif abs(va) > abs(ha) and ha > 0:
                        if av == 0:
                            action_index = 6
                        else:
                            action_index = 7

                            # 카메라를 안움직이는 경우
                # 점프하는 경우
                elif (aj == 1):
                    if (af == 0):
                        action_index = 8
                    else:
                        action_index = 9

                # 앞으로 가는 경우
                elif (af == 1):
                    if (av == 0):
                        action_index = 10
                    else:
                        action_index = 11

                # 뒤로 가는 경우
                elif (ab == 1):
                    if (av == 0):
                        action_index = 12
                    else:
                        action_index = 13

                # 왼쪽으로 가는 경우
                elif (al == 1):
                    if (av == 0):
                        action_index = 14
                    else:
                        action_index = 15

                # 오른쪽으로 가는 경우
                elif (ar == 1):
                    if (av == 0):
                        action_index = 16
                    else:
                        action_index = 17

                # 카메라, 움직임이 다 0이고 공격만 하는 것
                else:
                    if (av == 0):
                        continue
                    else:
                        action_index = 18

                a_index = torch.LongTensor([action_index])
                curr_obs = converter2(s_batch['pov'][i][j])
                _obs = converter2(ns_batch['pov'][i][j])
                _reward = torch.FloatTensor([r_batch[i][j]])
                _done = d_batch[i][j]  # .astype(int)

                n_step_state_buffer.append(curr_obs)
                n_step_action_buffer.append(a_index)
                n_step_reward_buffer.append(_reward)
                n_step_next_state_buffer.append(_obs)
                n_step_done_buffer.append(_done)
                n_rewards = sum([gamma * reward for gamma, reward in zip(gamma_list, n_step_reward_buffer)])
                n_step_n_rewards_buffer.append(n_rewards)


                rep_buffer.put((n_step_state_buffer[0], n_step_action_buffer[0], n_step_reward_buffer[0], n_step_next_state_buffer[0], n_step_done_buffer[0], n_step_n_rewards_buffer[0]))
                episode_start_ts += 1
                parse_ts += 1

                # if episode done we reset
                if _done:
                    break

        # replay is over emptying the deques
        if rep_buffer.size() > rep_buffer.buffer_limit:
            rep_buffer.buffer.popleft()
        print('Parse finished. {} expert samples added.'.format(parse_ts))
    print('add_demo finished')
    return rep_buffer

def append_sample(memory, model, target_model, state, action, reward, next_state, done, n_rewards):
    target = model(state).data.cpu()
    old_val = target[0][action]
    target_val = target_model(next_state).cpu()
    if done:
        target[0][action] = reward
    else:
        target[0][action] = reward + 0.99 * torch.max(target_val)

    error = abs(old_val - target[0][action])
    error = error.detach()
    memory.add.remote(error, (state, action, reward, next_state, done, n_rewards))

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

def pre_train(env_name, rep_buffer, policy_net, target_net, optimizer, threshold=10, num_epochs=1, batch_size=16,
              seq_len=10, gamma=0.99, model_name='pretrained', nstep=10):
    data = minerl.data.make(env_name)
    print("data loading sucess")
    demo_num = 0
    for s_batch, a_batch, r_batch, ns_batch, d_batch in data.batch_iter(num_epochs=num_epochs, batch_size=batch_size,
                                                                        seq_len=seq_len):
        demo_num += 1
        print(demo_num)
        print(r_batch.sum())
        if r_batch.sum() < threshold:
            continue
        """
        state_batch : (batch_size, seq_len, 64, 64, 3)
        action_batch : (batch_size, seq_len, action['property'].shape) ex camera = 2 otherwise 1

        reward_batch : (batch_size, seq_len)
        next_state_batch : (batch_size, seq_len, 64, 64, 3)
        done_batch : (batch_size, seq_len)


        reward, _ = stats.mode(r_batch, axis=1)
        reward = np.squeeze(reward)
        done, _ = stats.mode(d_batch, axis=1)
        done = np.squeeze(done)
        """
        parse_ts = 0

        batch_length = (s_batch['pov'].shape)[0]  # (batch, seq, 64, 64, 3)[0]
        for i in range(0, batch_length):
            episode_start_ts = 0

            n_step = nstep
            n_step_state_buffer = deque(maxlen=n_step)
            n_step_action_buffer = deque(maxlen=n_step)
            n_step_reward_buffer = deque(maxlen=n_step)
            n_step_n_rewards_buffer = deque(maxlen=n_step)
            n_step_next_state_buffer = deque(maxlen=n_step)
            n_step_done_buffer = deque(maxlen=n_step)
            gamma_list = [gamma ** i for i in range(n_step)]

            for j in range(0, seq_len):
                av = a_batch['attack'][i][j]  # attack value
                aj = a_batch['jump'][i][j]  # jump value
                af = a_batch['forward'][i][j]  # forward value
                ab = a_batch['back'][i][j]  # back value
                al = a_batch['left'][i][j]  # left value
                ar = a_batch['right'][i][j]  # right value
                va = a_batch['camera'][i][j][0]  # vertical angle and
                ha = a_batch['camera'][i][j][1]  # horizontal angle

                camera_thresholds = (abs(va) + abs(ha)) / 2.0
                # 카메라를 움직이는 경우
                if (camera_thresholds > 2.5):
                    # camera = [0, -5]
                    if abs(va) < abs(ha) and ha < 0:
                        if av == 0:
                            action_index = 0
                        else:
                            action_index = 1
                    # camera = [0, 5]
                    elif abs(va) < abs(ha) and ha > 0:
                        if av == 0:
                            action_index = 2
                        else:
                            action_index = 3
                    # camera = [-5, 0]
                    elif abs(va) > abs(ha) and ha < 0:
                        if av == 0:
                            action_index = 4
                        else:
                            action_index = 5
                    # camera = [5, 0]
                    elif abs(va) > abs(ha) and ha > 0:
                        if av == 0:
                            action_index = 6
                        else:
                            action_index = 7

                            # 카메라를 안움직이는 경우
                # 점프하는 경우
                elif (aj == 1):
                    if (af == 0):
                        action_index = 8
                    else:
                        action_index = 9

                # 앞으로 가는 경우
                elif (af == 1):
                    if (av == 0):
                        action_index = 10
                    else:
                        action_index = 11

                # 뒤로 가는 경우
                elif (ab == 1):
                    if (av == 0):
                        action_index = 12
                    else:
                        action_index = 13

                # 왼쪽으로 가는 경우
                elif (al == 1):
                    if (av == 0):
                        action_index = 14
                    else:
                        action_index = 15

                # 오른쪽으로 가는 경우
                elif (ar == 1):
                    if (av == 0):
                        action_index = 16
                    else:
                        action_index = 17

                # 카메라, 움직임이 다 0이고 공격만 하는 것
                else:
                    if (av == 0):
                        pass
                    else:
                        action_index = 18

                a_index = torch.LongTensor([action_index]).cpu()
                curr_obs = converter2(s_batch['pov'][i][j]).float().cpu()
                _obs = converter2(ns_batch['pov'][i][j]).float().cpu()
                _reward = torch.FloatTensor([r_batch[i][j]]).cpu()
                _done = d_batch[i][j]  # .astype(int)

                n_step_state_buffer.append(curr_obs)
                n_step_action_buffer.append(a_index)
                n_step_reward_buffer.append(_reward)
                n_step_next_state_buffer.append(_obs)
                n_step_done_buffer.append(_done)
                n_rewards = sum([gamma * reward for gamma, reward in zip(gamma_list, n_step_reward_buffer)])
                n_step_n_rewards_buffer.append(n_rewards)

                append_sample(rep_buffer, policy_net, target_net, n_step_state_buffer[-1], \
                              n_step_action_buffer[-1], n_step_reward_buffer[-1], \
                              n_step_next_state_buffer[-1], \
                              n_step_done_buffer[-1], \
                              n_step_n_rewards_buffer[-1])
                episode_start_ts += 1
                parse_ts += 1
                # if episode done we reset
                if _done:
                    break

        # replay is over emptying the deques
        # if rep_buffer.size() > rep_buffer.buffer_limit:
        #    rep_buffer.buffer.popleft()
        print('Parse finished. {} expert samples added.'.format(parse_ts))
        train_dqn(policy_net, target_net, rep_buffer, 256, optimizer)
        torch.save(policy_net.state_dict(), model_path + model_name)
        if demo_num % 5 == 0 and demo_num != 0:
            # 특정 반복 수가 되면 타겟 네트워크도 업데이트
            print("target network updated")
            target_net.load_state_dict(policy_net.state_dict())
        print("train {} step finished".format(demo_num))
    print('pre_train finished')
    return rep_buffer

@ray.remote
def parse_demo2(env_name, rep_buffer, policy_net, target_net, threshold=10, num_epochs=1, batch_size=16,
              seq_len=10, gamma=0.99):
    data = minerl.data.make(env_name)
    print("data loading sucess")
    demo_num = 0
    for s_batch, a_batch, r_batch, ns_batch, d_batch in data.batch_iter(num_epochs=num_epochs, batch_size=batch_size,seq_len=seq_len):
        if(ray.get(rep_buffer.size.remote()) > 10000):
            break

        demo_num += 1
        print(demo_num)
        print(r_batch.sum())
        if r_batch.sum() < threshold:
            del s_batch, a_batch, r_batch, d_batch, ns_batch
            continue
        """
        state_batch : (batch_size, seq_len, 64, 64, 3)
        action_batch : (batch_size, seq_len, action['property'].shape) ex camera = 2 otherwise 1

        reward_batch : (batch_size, seq_len)
        next_state_batch : (batch_size, seq_len, 64, 64, 3)
        done_batch : (batch_size, seq_len)


        reward, _ = stats.mode(r_batch, axis=1)
        reward = np.squeeze(reward)
        done, _ = stats.mode(d_batch, axis=1)
        done = np.squeeze(done)
        """
        parse_ts = 0

        batch_length = (s_batch['pov'].shape)[0]  # (batch, seq, 64, 64, 3)[0]
        for i in range(0, batch_length):
            episode_start_ts = 0

            n_step = 10
            n_step_state_buffer = deque(maxlen=n_step)
            n_step_action_buffer = deque(maxlen=n_step)
            n_step_reward_buffer = deque(maxlen=n_step)
            n_step_n_rewards_buffer = deque(maxlen=n_step)
            n_step_next_state_buffer = deque(maxlen=n_step)
            n_step_done_buffer = deque(maxlen=n_step)
            gamma_list = [gamma ** i for i in range(n_step)]

            for j in range(0, seq_len):
                av = a_batch['attack'][i][j]  # attack value
                aj = a_batch['jump'][i][j]  # jump value
                af = a_batch['forward'][i][j]  # forward value
                ab = a_batch['back'][i][j]  # back value
                al = a_batch['left'][i][j]  # left value
                ar = a_batch['right'][i][j]  # right value
                va = a_batch['camera'][i][j][0]  # vertical angle and
                ha = a_batch['camera'][i][j][1]  # horizontal angle

                camera_thresholds = (abs(va) + abs(ha)) / 2.0
                # 카메라를 움직이는 경우
                if (camera_thresholds > 2.5):
                    # camera = [0, -5]
                    if abs(va) < abs(ha) and ha < 0:
                        if av == 0:
                            action_index = 0
                        else:
                            action_index = 1
                    # camera = [0, 5]
                    elif abs(va) < abs(ha) and ha > 0:
                        if av == 0:
                            action_index = 2
                        else:
                            action_index = 3
                    # camera = [-5, 0]
                    elif abs(va) > abs(ha) and ha < 0:
                        if av == 0:
                            action_index = 4
                        else:
                            action_index = 5
                    # camera = [5, 0]
                    elif abs(va) > abs(ha) and ha > 0:
                        if av == 0:
                            action_index = 6
                        else:
                            action_index = 7

                            # 카메라를 안움직이는 경우
                # 점프하는 경우
                elif (aj == 1):
                    if (af == 0):
                        action_index = 8
                    else:
                        action_index = 9

                # 앞으로 가는 경우
                elif (af == 1):
                    if (av == 0):
                        action_index = 10
                    else:
                        action_index = 11

                # 뒤로 가는 경우
                elif (ab == 1):
                    if (av == 0):
                        action_index = 12
                    else:
                        action_index = 13

                # 왼쪽으로 가는 경우
                elif (al == 1):
                    if (av == 0):
                        action_index = 14
                    else:
                        action_index = 15

                # 오른쪽으로 가는 경우
                elif (ar == 1):
                    if (av == 0):
                        action_index = 16
                    else:
                        action_index = 17

                # 카메라, 움직임이 다 0이고 공격만 하는 것
                else:
                    if (av == 0):
                        continue
                    else:
                        action_index = 18

                a_index = torch.LongTensor([action_index]).cpu()
                curr_obs = converter2(s_batch['pov'][i][j]).float().cpu()
                _obs = converter2(ns_batch['pov'][i][j]).float().cpu()
                _reward = torch.FloatTensor([r_batch[i][j]]).cpu()
                _done = d_batch[i][j]  # .astype(int)

                n_step_state_buffer.append(curr_obs)
                n_step_action_buffer.append(a_index)
                n_step_reward_buffer.append(_reward)
                n_step_next_state_buffer.append(_obs)
                n_step_done_buffer.append(_done)
                n_rewards = sum([gamma * reward for gamma, reward in zip(gamma_list, n_step_reward_buffer)])
                n_step_n_rewards_buffer.append(n_rewards)

                append_sample(rep_buffer, policy_net, target_net, n_step_state_buffer[-1], \
                              n_step_action_buffer[-1], n_step_reward_buffer[-1], \
                              n_step_next_state_buffer[-1], \
                              n_step_done_buffer[-1], \
                              n_step_n_rewards_buffer[-1])
                episode_start_ts += 1
                parse_ts += 1
                # if episode done we reset
                if _done:
                    break

            del n_step_state_buffer
            del n_step_action_buffer
            del n_step_reward_buffer
            del n_step_n_rewards_buffer
            del n_step_next_state_buffer
            del n_step_done_buffer
        # replay is over emptying the deques
        # if rep_buffer.size() > rep_buffer.buffer_limit:
        #    rep_buffer.buffer.popleft()
        print('Parse finished. {} expert samples added.'.format(parse_ts))


def save(model, out_dir=None):
    torch.save(model.state_dict, out_dir)

def load(model, out_dir=None):
    model.load_state_dict(torch.load(out_dir))
    model.eval()
