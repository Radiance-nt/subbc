import argparse
import time

import numpy as np
import gym
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from model.net import Net
from utils import to_input

plt.style.use("ggplot")

def train_BC(agent, dataset, args):
    '''
    train Behavioral Cloning model, given pair of states return action (s0,s1 ---> a0 if n=2)
    Input:
    training_set:
    policy: Behavioral Cloning model want to train
    n: window size (how many states needed to predict the next action)
    batch_size: batch size
    n_epoch: number of epoches
    return:
    policy: trained Behavioral Cloning model
    '''

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    step = 0
    best_reward = None
    loss_his = []

    for batch in dataloader:
        obs, gold_actions = batch
        pred_actions = agent(obs)
        loss = loss_fn(pred_actions, gold_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_his.append(loss.item())

    return loss_his




def test_agent(env, model, args):
    seed_reward = []
    for itr in range(args.n_iterations):
        ################################## interact with env ##################################
        G = []
        for epoch in range(args.max_epoch):
            state = env.reset()
            rewards = []
            R = 0
            for t in range(args.max_steps):
                action = model(torch.tensor(state, dtype=torch.float))
                action = torch.argmax(action).item()
                # action = np.clip(action.detach().numpy(), -1, 1)
                next_state, r, done, _ = env.step(action)
                env.render()
                # time.sleep(0.01)
                rewards.append(r)
                state = next_state
                if done:
                    break
            R = sum([rewards[i] * args.gamma ** i for i in range(len(rewards))])
            G.append(R)
        seed_reward.append(G)
        print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward[-1])))
        print("Interacting with environment finished")
    print("Seed reward: ",seed_reward)
    seed_reward_mean_bc = np.array(seed_reward)
    mean_bc = np.mean(seed_reward_mean_bc, axis=0)
    std_bc = np.std(seed_reward_mean_bc, axis=0)
    return seed_reward





# np.save("reward_mean_walker_bc1_expert_states={}".format(new_data.shape[0]), seed_reward_mean) #uncomment to save reward over 5 random seeds




def main(args):
    env_name = 'MountainCar-v0'  # 'MountainCar-v0'
    env = gym.make(args.env)
    action_space_size = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 4
    state_space_size = env.observation_space.shape[0]
    expert_obs = torch.tensor(np.load("expert/data/%s_state_array.npy" % env_name), dtype=torch.float)
    expert_actions = torch.tensor(np.load("expert/data/%s_action_array.npy" % env_name), dtype=torch.float)

    expert_obs = torch.from_numpy(expert_obs).to(args.device)
    expert_actions = torch.from_numpy(expert_actions).to(args.device)
    dataset = TensorDataset(expert_obs, expert_actions)

    # a = np.random.randint(1 + expert_obs.shape[0] - number_expert_trajectories)

    net = Net(state_space_size, action_space_size)

    best_acc = 0
    for epoch in range(args.epochs):
        train_BC()
        test_acc = test_BC()
        if test_acc >= best_acc:
            best_acc = test_acc
        seed_reward = test_agent(env, net, n_iterations=5, n_ep=20)

    env.close()

    print('best accuracy:', best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains on GoBigger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--exp_name', type=str, default='gobigger_vsbot_eval', help='Root for the Cifar dataset.')
    parser.add_argument('--match_time', type=int, default=600)
    # Optimization options
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--actor_lr', type=float, default=0.004)
    parser.add_argument('--critic_lr', type=float, default=0.0002)

    parser.add_argument('--collector_env_num', type=int, default=10)
    parser.add_argument('--evaluator_env_num', type=int, default=0)
    parser.add_argument('--n_evaluator_episode', type=int, default=1)

    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--update_per_collect', type=int, default=20)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--step_per_collect', type=int, default=320)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument("--bc", default='False', action='store_true', help='Use behavior cloning or not')
    parser.add_argument("--noise", type=float, default='2', help='GaussianNoise noise rate')

    parser.add_argument('--ckpt', '-c', help='checkpoint for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_path', type=str,
                        default='./log/{}'.format(time.strftime("%b_%d_%H_%M", time.localtime())),
                        help='Root for the Cifar dataset.')

    args = parser.parse_args()

    main(args)
