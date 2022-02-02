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
    for itr in range(args.n_evaluator_episode):
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
        R = sum([rewards[i] * args.discount_factor ** i for i in range(len(rewards))])
        seed_reward.append(R)
    print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward[-1])))
    print("Interacting with environment finished")
    print("Seed reward: ", seed_reward)
    seed_reward_mean_bc = np.array(seed_reward)
    mean_bc = np.mean(seed_reward_mean_bc, axis=0)
    std_bc = np.std(seed_reward_mean_bc, axis=0)
    return seed_reward


def main(args):
    env = gym.make(args.env)
    action_space_size = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 4
    state_space_size = env.observation_space.shape[0]
    expert_obs = torch.tensor(np.load("expert/data/%s_state_array.npy" % args.env), dtype=torch.float)
    expert_actions = torch.tensor(np.load("expert/data/%s_action_array.npy" % args.env), dtype=torch.float)

    expert_obs = torch.from_numpy(expert_obs).to(args.device)
    expert_actions = torch.from_numpy(expert_actions).to(args.device)
    dataset = TensorDataset(expert_obs, expert_actions)

    # a = np.random.randint(1 + expert_obs.shape[0] - number_expert_trajectories)

    net = Net(state_space_size, action_space_size)

    best_acc = 0
    for epoch in range(args.epochs):
        train_BC()
        test_BC()
        seed_reward = test_agent(env, net, n_iterations=5, n_ep=20)

    env.close()

    print('best accuracy:', best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--env', type=str, default='MountainCar-v0', help='Env name')
    parser.add_argument('--match_time', type=int, default=600)
    # Optimization options
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--n_evaluator_episode', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--discount_factor', type=float, default=0.99)

    parser.add_argument('--ckpt', '-c', help='checkpoint for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_path', type=str,
                        default='./log/{}'.format(time.strftime("%b_%d_%H_%M", time.localtime())),
                        help='Root for the Cifar dataset.')

    args = parser.parse_args()

    main(args)
