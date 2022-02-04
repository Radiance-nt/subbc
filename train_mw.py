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
import torch.nn.functional as F
from model.net import Net, VNet
from utils import to_input

plt.style.use("ggplot")


def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, cfg):
    epoch = cfg['epoch']
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        loss_fn = F.mse_loss
        model.train()
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        meta_model = Net(cfg['state_space_size'], cfg['action_space_size']).to(args.device)
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(inputs)

        cost = loss_fn(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), -1)).mean(1, True)
        v_lambda = vnet(cost_v.data)
        l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))  # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(args.device), targets_val.to(args.device)
        y_g_hat = meta_model(inputs_val)
        l_g_meta = loss_fn(y_g_hat, targets_val)
        # prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        outputs = model(inputs)
        cost_w = loss_fn(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), -1)).mean(1, True)
        # prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)

        loss = torch.sum(cost_v * w_new) / len(cost_v)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()

        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                      (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1))))
        return train_loss, meta_loss


def train_BC(agent, dataloader, args):
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.l2)
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


def valid_BC(agent, dataloader, args):
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.l2)
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

    def load_data(sample, tag):
        expert_obs = np.load('expert/data/{}_{}_state_array.npy'.format(args.task, tag))
        expert_actions = np.load('expert/data/{}_{}_action_array.npy'.format(args.task, tag))
        expert_reward = np.load('expert/data/{}_{}_reward_array.npy'.format(args.task, tag))
        assert expert_obs.shape[0] == expert_actions.shape[0]
        assert expert_obs.shape[1] == expert_actions.shape[1]
        print('Loading tag: ', tag)
        expert_reward = expert_reward.mean(1)
        expert_mean = sum([expert_reward[i] * args.discount_factor ** i for i in range(len(expert_reward))])
        print('reward_mean: ', expert_mean)
        total_traj = expert_obs.shape[0]
        traj_length = expert_obs.shape[1]
        sample_num = sample * traj_length
        test_sample_num = int(sample_num / 10)
        print('Total traj: {} | Traj length: {} | Sample_traj: {}+{}'.format(
            total_traj, traj_length, sample, int(sample / 10)))

        expert_obs = expert_obs.reshape(-1, expert_obs.shape[-1])
        expert_actions = expert_actions.reshape(-1, expert_actions.shape[-1])

        expert_obs = torch.from_numpy(expert_obs).to(torch.float32)
        expert_actions = torch.from_numpy(expert_actions).to(torch.float32)

        train_dataset = TensorDataset(expert_obs[:sample_num], expert_actions[:sample_num])
        test_dataset = TensorDataset(expert_obs[sample_num:sample_num + test_sample_num],
                                     expert_actions[sample_num:sample_num + test_sample_num])

        return train_dataset, test_dataset

    train_meta_dataset, test_dataset = load_data(args.sample, 'gold')
    train_dataset, _ = load_data(args.sample * 2, 'silver')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_meta_loader = DataLoader(train_meta_dataset, batch_size=args.batch_size, shuffle=True)

    # a = np.random.randint(1 + expert_obs.shape[0] - number_expert_trajectories)

    agent = Net(state_space_size, action_space_size).to(args.device)
    vnet = VNet(1, 100, 1).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.l2)
    optimizer_vnet = optim.Adam(vnet.parameters(), 1e-3,
                                weight_decay=1e-4)
    writer = SummaryWriter(os.path.join(
        args.log_path, args.task, 'hybrid' + '_' + str(args.sample)))
    for epoch in range(args.epoch):
        cfg = {'epoch': epoch, 'action_space_size': action_space_size, 'state_space_size': state_space_size}
        train_loss, meta_loss = train(train_loader, train_meta_loader,
                                      agent, vnet, optimizer, optimizer_vnet, cfg)
        valid_loss = valid_BC(agent, test_dataset, args)
        # test_BC()
        _, r_mean, r_std = test_agent(env, agent, args)
        writer.add_scalar('phase/train_loss', train_loss, epoch)
        writer.add_scalar('phase/meta_loss', meta_loss, epoch)
        writer.add_scalar('phase/valid_loss', valid_loss, epoch)
        writer.add_scalar('phase/r_mean', r_mean, epoch)
        writer.add_scalar('phase/r_std', r_std, epoch)
        # writer.add_scalar('phase/r_expert', expert_mean, epoch)
        print('Epoch: {} | Validation loss: {} | Reward : {}'.format(epoch, valid_loss, r_mean))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--task', type=str, default='Ant-v3', help='Env name')
    parser.add_argument('--sample', type=int, default=50)

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
