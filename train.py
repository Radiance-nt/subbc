import argparse
import os
import time

import numpy as np
import gym
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.net import Net
from utils import to_input

plt.style.use("ggplot")


def train_BC(agent, dataset, args):
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.l2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    loss_sum = 0
    for i, batch in enumerate(dataloader):
        obs, gold_actions = batch
        if args.device == 'cuda':
            obs, gold_actions = obs.cuda(), gold_actions.cuda()
        pred_actions = agent(obs)
        loss = loss_fn(pred_actions, gold_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    loss_avg = loss_sum / i
    return loss_avg


def valid_BC(agent, dataset, args):
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.l2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    loss_fn = nn.MSELoss()
    loss_sum = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            obs, gold_actions = batch
            if args.device == 'cuda':
                obs, gold_actions = obs.cuda(), gold_actions.cuda()
            pred_actions = agent(obs)
            loss = loss_fn(pred_actions, gold_actions)
            loss_sum += loss.item()
        loss_avg = loss_sum / i
    return loss_avg


def test_agent(env, model, args):
    seed_reward = []
    with torch.no_grad():
        for itr in range(args.n_evaluator_episode):
            state = env.reset()
            rewards = []
            R = 0
            for t in range(args.max_steps):
                state = torch.tensor(state, dtype=torch.float).to(args.device)
                action = model(state)
                action = action.cpu().numpy()
                state, r, done, _ = env.step(action)
                rewards.append(r)
                if done:
                    break
            R = sum([rewards[i] * args.discount_factor ** i for i in range(len(rewards))])
            seed_reward.append(R)
    seed_reward_mean_bc = np.array(seed_reward)
    mean_bc = np.mean(seed_reward_mean_bc, axis=0)
    std_bc = np.std(seed_reward_mean_bc, axis=0)
    return seed_reward_mean_bc, mean_bc, std_bc


def main(args):
    env = gym.make(args.task)
    action_space_size = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 4
    state_space_size = env.observation_space.shape[0]
    print('action_space_size: ', action_space_size)
    print('state_space_size: ', state_space_size)

    expert_obs = np.load('expert/data/{}_{}_state_array.npy'.format(args.task, args.tag))
    expert_actions = np.load('expert/data/{}_{}_action_array.npy'.format(args.task, args.tag))
    expert_reward = np.load('expert/data/{}_{}_reward_array.npy'.format(args.task, args.tag))

    expert_reward = expert_reward.mean(1)
    expert_mean = sum([expert_reward[i] * args.discount_factor ** i for i in range(len(expert_reward))])
    print('expert_mean: ', expert_mean)

    assert expert_obs.shape[0] == expert_actions.shape[0]
    assert expert_obs.shape[1] == expert_actions.shape[1]
    total_traj = expert_obs.shape[0]
    traj_length = expert_obs.shape[1]
    sample_num = args.sample * traj_length
    test_sample_num = int(sample_num / 10)
    print('Total traj: {} | Traj length: {} | Sample_traj: {}+{}'.format(
        total_traj, traj_length, args.sample, int(args.sample / 10)))

    expert_obs = expert_obs.reshape(-1, expert_obs.shape[-1])
    expert_actions = expert_actions.reshape(-1, expert_actions.shape[-1])

    expert_obs = torch.from_numpy(expert_obs).to(torch.float32)
    expert_actions = torch.from_numpy(expert_actions).to(torch.float32)


    train_dataset = TensorDataset(expert_obs[:sample_num], expert_actions[:sample_num])
    test_dataset = TensorDataset(expert_obs[sample_num:sample_num+test_sample_num],
                                 expert_actions[sample_num:sample_num+test_sample_num])

    writer = SummaryWriter(os.path.join(
        args.log_path, args.task, args.tag + '_' + str(args.sample)))
    # a = np.random.randint(1 + expert_obs.shape[0] - number_expert_trajectories)

    agent = Net(state_space_size, action_space_size)
    agent.to(args.device)


    for epoch in range(args.epoch):
        train_loss = train_BC(agent, train_dataset, args)
        valid_loss = valid_BC(agent, test_dataset, args)
        # test_BC()
        _, r_mean, r_std = test_agent(env, agent, args)
        writer.add_scalar('phase/train_loss', train_loss, epoch)
        writer.add_scalar('phase/valid_loss', valid_loss, epoch)
        writer.add_scalar('phase/r_mean', r_mean, epoch)
        writer.add_scalar('phase/r_std', r_std, epoch)
        writer.add_scalar('phase/r_expert', expert_mean, epoch)
        print('Epoch: {} | Validation loss: {} | Reward : {}'.format(epoch, valid_loss, r_mean))

    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--task', type=str, default='Ant-v3', help='Env name')
    parser.add_argument('--sample', type=int, default=50)
    parser.add_argument('--tag', type=str, default='gold')

    # Optimization options
    parser.add_argument('--epoch', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.0001)

    parser.add_argument('--n_evaluator_episode', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--discount_factor', type=float, default=0.99)

    parser.add_argument('--ckpt', '-c', help='checkpoint for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_path', type=str,
                        default='./log', help='Root for the Cifar dataset.')

    args = parser.parse_args()

    main(args)
