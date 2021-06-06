@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, network, batch_size):
        self.learner_network = DQN(19).cuda().float()
        self.learner_target_network = DQN(19).cuda().float()
        self.learner_network.load_state_dict(network.state_dict())
        self.learner_target_network.load_state_dict(network.state_dict())
        self.shared_network = DQN(19).cpu()
        self.shared_target_network = DQN(19).cpu()

        self.count = 0
        self.batch_size = batch_size
        self.max_counts = 1000000

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
        print("return learner network")
        return self.shared_network

    def get_target_network(self):
        self.shared_target_network.load_state_dict(self.learner_target_network.state_dict())
        return self.shared_target_network

    def update_network(self, memory, demos, batch_size, optimizer):
        print("started")
        print(ray.get(memory.size.remote()))
        print(ray.get(demos.size.remote()))

        counts = [x for x in range(self.max_counts)]
        train_stats = pd.DataFrame(index=counts, columns=['loss'])
        while (self.count < 1000000):

            agent_batch, agent_idxs, agent_weights = ray.get(memory.sample.remote(batch_size))
            demo_batch, demo_idxs, demo_weights = ray.get(demos.sample.remote(batch_size))

            # demo_batch = (batch_size, state, action, reward, next_state, done, n_rewards)
            # print(len(demo_batch[0])) # 0번째 배치이므로 0이 나옴
            state_list = []
            action_list = []
            reward_list = []
            next_state_list = []
            done_mask_list = []

            for agent_transition in agent_batch:
                s, a, r, s_prime, done_mask, = agent_transition
                state_list.append(s)
                action_list.append([a])
                reward_list.append([r])
                next_state_list.append(s_prime)
                done_mask_list.append([done_mask])

            for expert_transition in demo_batch:
                s, a, r, s_prime, done_mask = expert_transition
                state_list.append(s)
                action_list.append([a])
                reward_list.append([r])
                next_state_list.append(s_prime)
                done_mask_list.append([done_mask])

            s = torch.stack(state_list).float().cuda()
            a = torch.tensor(action_list, dtype=torch.int64).cuda()
            r = torch.tensor(reward_list).cuda()
            s_prime = torch.stack(next_state_list).float().cuda()
            done_mask = torch.tensor(done_mask_list).float().cuda()

            q_vals = self.learner_network(s)
            state_action_values = q_vals.gather(1, a)

            # comparing the q values to the values expected using the next states and reward
            next_state_values = self.learner_target_network(s_prime).max(1)[0].unsqueeze(1)
            target = r + (next_state_values * gamma * done_mask)

            # calculating the q loss, n-step return lossm supervised_loss
            is_weights = torch.FloatTensor(agent_weights).to(device)
            q_loss = (is_weights * F.mse_loss(state_action_values, target)).mean()
            # supervised_loss = margin_loss(q_vals, a, 1, 1)

            loss = q_loss  # + supervised_loss
            errors = torch.abs(state_action_values - target).data.cpu().detach()
            errors = errors.numpy()
            # update priority
            for i in range(batch_size):
                idx = agent_idxs[i]
                memory.update.remote(idx, errors[i])

            train_stats.loc[self.count]['loss'] = float(loss.item())
            train_stats.to_csv('train_stat_minerl_learner.csv')

            # optimization step and logging
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(self.learner_network.state_dict(), model_path + "apex_dqfd_learner.pth")
            self.count += 1
            if (self.count % 100 == 0 and self.count != 0):
                self.learner_target_network.load_state_dict(self.learner_network.state_dict())
                print("Count : {} leaner_target_network updated".format(self.count))

            if (self.count % 10 == 0 and self.count != 0):
                print("Count : {} leaner_network updated".format(self.count))



