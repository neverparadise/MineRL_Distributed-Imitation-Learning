from model import *
from utils import *
from replay_buffer import Memory, RemoteMemory
import argparse
import os
import yaml
import pytest

#하이퍼 파라미터
with open('navigate.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser(description='argparse for pretraining')
parser.add_argument('--model_name', type=str, default="pre_trained", help='pre_trained model name')
parse = parser.parse_args()

model_name = parse.model_name
config = {'model_name': parse.model_name}

env_name = args['env_name']
gamma = args['gamma']
learning_rate = args['lr']
# max_epi = args['max_epi']
# agent_num = args['agent_num']
num_gpus = args['num_gpus']
num_channels = args['num_channels']
buffer_limit = args['buffer_limit']
L1 = args['L1']
pretrain_epochs = args['pretrain_epochs']
threshold = args['threshold']
batch_size = args['batch_size']
seq_len = args['seq_len']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.curdir + '/trained_model/'

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

def train_dqn(policy_net, target_net, memory, batch_size, optimizer):
    batch, idxs, is_weights = memory.sample(batch_size)
    state_list = []
    action_list = []
    reward_list = []
    next_state_list = []
    done_mask_list = []
    n_rewards_list = []

    for i, transition in enumerate(batch):
        s, a, r, s_prime, done_mask, n_rewards = transition
        s = torch.tensor(s).float().to(device)
        a = torch.tensor([a], dtype=torch.int64).to(device)
        r = torch.tensor([r]).to(device)
        s_prime = torch.tensor(s_prime).float().to(device)
        done_mask = torch.tensor(done_mask).float().to(device)
        nr = torch.tensor(n_rewards).to(device)


        q_vals = policy_net(s)
        print(q_vals)
        print(q_vals.shape)
        state_action_values = q_vals.gather(1, a)

        # comparing the q values to the values expected using the next states and reward
        next_state_values = target_net(s_prime).max(1)[0].unsqueeze(1)
        target = r + (next_state_values * gamma) * done_mask

        # calculating the q loss, n-step return lossm supervised_loss
        is_weight = torch.FloatTensor(is_weights[i]).to(device)
        q_loss = (is_weight * F.mse_loss(state_action_values, target)).mean()
        n_step_loss = (state_action_values.max(1)[0] + nr).mean()
        supervised_loss = margin_loss(q_vals, a, 1, 1)

        loss = q_loss + supervised_loss + n_step_loss + F.l1_loss(state_action_values, target)

        errors = torch.abs(state_action_values - target).data.cpu()
        errors = errors.numpy()
        # update priority
        idx = idxs[i]
        memory.update(idx, errors[i])
        # optimization step and logging
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def pre_train(env_name, rep_buffer, policy_net, target_net, optimizer):
    model_name = 'pre_trained_dqn.pth'
    gamma = args['gamma']
    num_epochs = args['pretrain_epochs']
    threshold = args['threshold']
    batch_size = args['batch_size']
    seq_len = args['seq_len']

    # Data loading
    os.environ['MINERL_DATA_ROOT'] = '/home/neverparadise/MineRL_DATA/data_texture_0_low_res'
    data = minerl.data.make(env_name)
    print("data loading sucess")
    demo_num = 0
    for s_batch, a_batch, r_batch, ns_batch, d_batch in data.batch_iter(num_epochs=num_epochs, batch_size=batch_size,
                                                                        seq_len=seq_len):
        demo_num += 1
        if demo_num % 10 == 0:
            print(f"demo_num : {demo_num}")
            print(f"r sum : {r_batch.sum()}")

        if demo_num % 500 == 0:
            torch.save(policy_net.state_dict(), model_path + str(demo_num) + model_name)
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
                action_index = 18
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
                if env_name == 'MineRLTreechop-v0':
                    curr_obs = converter_for_pretrain(env_name, s_batch['pov'][i][j]).float().cpu()
                    _obs = converter_for_pretrain(env_name, ns_batch['pov'][i][j]).float().cpu()
                else:
                    curr_obs = converter_for_pretrain(env_name, s_batch['pov'][i][j], s_batch['compassAngle'][i][j]).float().cpu()
                    _obs = converter_for_pretrain(env_name, ns_batch['pov'][i][j], ns_batch['compassAngle'][i][j]).float().cpu()
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
        train_dqn(policy_net, target_net, rep_buffer, batch_size, optimizer)
        if demo_num % 5 == 0 and demo_num != 0:
            # 특정 반복 수가 되면 타겟 네트워크도 업데이트
            print("target network updated")
            target_net.load_state_dict(policy_net.state_dict())
        print("train {} step finished".format(demo_num))
    print('pre_train finished')
    return rep_buffer

def save(model, out_dir=None):
    torch.save(model.state_dict, out_dir)

def load(model, dir=None):
    model.load_state_dict(torch.load(dir))
    model.eval()

def main():
    policy_net = DQN(num_channels=num_channels, num_actions=19).to(device=device)
    target_net = DQN(num_channels=num_channels, num_actions=19).to(device=device)
    target_net.load_state_dict(policy_net.state_dict())
    memory = Memory(50000)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
    print("pre_train start")
    model_name = 'pre_trained_dqn'
    pre_train(env_name, memory, policy_net, target_net, optimizer)
    print("pre_train finished")

main()