#!/usr/bin/env python3
import sys
import time

sys.path.append('.')
import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, Batch
from tianshou.env import SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v3')
    parser.add_argument('--tag', type=str, default='gold')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    return parser.parse_args()


def test_td3(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.policy_noise = args.policy_noise * args.max_action
    args.noise_clip = args.noise_clip * args.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # train_envs = gym.make(args.task)

    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(
        net_a, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = TD3Policy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step,
        action_space=env.action_space
    )

    if not args.watch:
        pass
    else:
        policy.load_state_dict(torch.load(
            os.path.join('policy', '{}_{}.pth'.format(args.task, args.tag))))
    # Let's watch its performance!
    policy.eval()

    test_episode = 2000
    max_step = 1000
    state_array = np.zeros((test_episode, max_step, np.prod(args.state_shape)))
    action_array = np.zeros((test_episode, max_step, np.prod(args.action_shape)))
    reward_array = np.zeros((test_episode, max_step))

    returns = []
    total_observations = []
    total_actions = []
    with torch.no_grad():
        for i_episode in range(test_episode):
            obs = env.reset()
            observations = []
            actions = []
            done = False
            totalr = 0
            steps = 0
            while not done:
                result = policy(Batch({'obs': obs[np.newaxis, :], 'info': ''}))
                action = result.act.cpu().numpy()
                action = action.reshape(-1)

                state_array[i_episode,  steps] = obs
                action_array[i_episode, steps] = action

                # observations.append(obs)
                # actions.append(action)
                obs, r, done, _ = env.step(action)
                # time.sleep(0.01)
                # env.render()
                reward_array[i_episode, steps] = r
                totalr += r
                steps += 1
                if steps >= max_step:
                    break
            returns.append(totalr)
            total_observations.append(observations)
            total_actions.append(actions)
            print('{} episode collected, total reward: {}'.format(i_episode, 0))
    np.save('data/{}_{}_state_array.npy'.format(args.task, args.tag), state_array)
    np.save('data/{}_{}_action_array.npy'.format(args.task, args.tag), action_array)
    np.save('data/{}_{}_reward_array.npy'.format(args.task, args.tag), reward_array)
    # avrg_mean, avrg_std = np.mean(returns), np.std(returns)
    # observations = np.array(observations).astype(np.float32)
    # actions = np.array(actions).astype(np.float32)
    # print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

    print(f'Finished')


if __name__ == '__main__':
    test_td3()
