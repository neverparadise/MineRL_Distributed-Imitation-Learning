
import ray
import torch
import torch.nn.functional as F

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


def train_drqn():
