import torch
import random
from replay_buffer import append_sample
from _collections import deque
import gym
import minerl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import ray
import os
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('navigate.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
env_name = args['env_name']

def make_6action(env, action_index):
    action = env.action_space.noop()
    if action_index == 0:
        action['forward'] = 1
    elif action_index == 1:
        action['jump'] = 1
    elif action_index == 2:
        action['camera'] = [0, -5]
    elif action_index == 3:
        action['camera'] = [0, 5]
    elif action_index == 4:
        action['camera'] = [-5, 0]
    elif action_index == 5:
        action['camera'] = [5, 0]

    return action


def make_19action(env, action_index):
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

def make_9action(env, action_index):
    # Action들을 정의
    action = env.action_space.noop()
    if (action_index == 0):
        action['camera'] = [0, -5]
        action['attack'] = 1
    elif (action_index == 1):
        action['camera'] = [0, 5]
        action['attack'] = 1
    elif (action_index == 2):
        action['camera'] = [-5, 0]
        action['attack'] = 1
    elif (action_index == 3):
        action['camera'] = [5, 0]
        action['attack'] = 1
    elif (action_index == 4):
        action['forward'] = 1
        action['jump'] = 1
    elif (action_index == 5):
        action['forward'] = 1
        action['attack'] = 1
    elif (action_index == 6):
        action['back'] = 1
        action['attack'] = 1
    elif (action_index == 7):
        action['left'] = 1
        action['attack'] = 1
    elif (action_index == 8):
        action['right'] = 1
        action['attack'] = 1

    return action

def parse_action(camera_angle):
    pass


def converter(env_name, observation):
    if (env_name == 'MineRLNavigateDense-v0' or
            env_name == 'MineRLNavigate-v0'):
        obs = observation['pov']
        obs = obs / 255.0  # [64, 64, 3]
        compass_angle = observation['compass']['angle']
        compass_angle_scale = 180
        compass_scaled = compass_angle / compass_angle_scale
        compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
        obs = np.concatenate([obs, compass_channel], axis=-1)
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        if(len(obs.shape) < 4):
            obs = obs.unsqueeze(0).to(device=device)
        return obs.float() # return (1, 4, 64, 64)
    else:
        obs = observation['pov']
        obs = obs / 255.0
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        if(len(obs.shape) < 4):
            obs = obs.unsqueeze(0).to(device=device)
        return obs.float() # return (1, 4, 64, 64)

def converter_for_pretrain(env_name, pov, compassAngle=None):
    if (env_name == 'MineRLNavigateDense-v0' or
            env_name == 'MineRLNavigate-v0'):
        obs = pov
        obs = obs / 255.0  # [64, 64, 3]
        compass_angle = compassAngle
        compass_angle_scale = 180
        compass_scaled = compass_angle / compass_angle_scale
        compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
        obs = np.concatenate([obs, compass_channel], axis=-1)
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        return obs.float()
    else:
        obs = pov
        obs = obs / 255.0
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        return obs.float()

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
                action_index = None
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
                current_obs = converter(env_name, s_batch['pov'][i][j]).float().cpu()
                next_obs = converter(env_name, ns_batch['pov'][i][j]).float().cpu()
                _reward = torch.FloatTensor([r_batch[i][j]]).cpu()
                _done = d_batch[i][j]  # .astype(int)

                n_step_state_buffer.append(current_obs)
                n_step_action_buffer.append(a_index)
                n_step_reward_buffer.append(_reward)
                n_step_next_state_buffer.append(next_obs)
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


