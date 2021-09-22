from model import DQN
from utils import *
from replay_buffer import Memory, RemoteMemory
import argparse
import os

#하이퍼 파라미터
learning_rate = 0.0003
gamma = 0.999
buffer_limit = 50000
L1 = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.curdir + '/dqn_model/'

def append_sample(memory, model, target_model, state, action, reward, next_state, done):
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
    if isinstance(memory, Memory):
        memory.add(error, [state, action, reward, next_state, done])
    else:
        memory.remote.add(error, [state, action, reward, next_state, done])

def pre_train(env_name, rep_buffer, policy_net, target_net, optimizer, threshold=10, num_epochs=1, batch_size=16,
              seq_len=10, gamma=0.99, model_name='pre_trained'):
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


parser = argparse.ArgumentParser(description='argparse for pretraining')
parser.add_argument('--env', type=str, default="MineRLTreechop-v0", help='name of MineRLEnvironment')
parser.add_argument('--threshold', type=int, default=640, help='reward threshold for parsing demos')
parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs')
parser.add_argument('--seq_len', type=int, default=400, help='sequence length for parsing demos')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma for calculating return')
parser.add_argument('--model_name', type=str, default="pre_trained", help='pre_trained model name')
args = parser.parse_args()

config = {'env  ': args.env,
          'threshold': args.threshold,
          'epochs': args.epochs,
          'batch_size': args.batch_size,
          'seq_len': args.seq_len,
          'gamma': args.gamma,
          'model_name': args.model_name}


def main():
    policy_net = DQN(19).to(device=device)
    target_net = DQN(19).to(device=device)
    target_net.load_state_dict(policy_net.state_dict())
    memory = Memory(50000)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
    print("pre_train start")
    pre_train(memory, policy_net, target_net, optimizer, **config)
    print("pre_train finished")