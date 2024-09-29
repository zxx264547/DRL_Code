# ========== 引用所需的库 ==========
import collections
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy
from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# 自定义模块导入
from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from IEEE_13_3p import IEEE13bus3p, create_13bus3p
from safeDDPG import ValueNetwork, SafePolicyNetwork, DDPG, ReplayBuffer, ReplayBufferPI, PolicyNetwork, \
    SafePolicy3phase, LinearPolicy

# ========== GPU 设置 ==========
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# ========== 参数设置 ==========
parser = argparse.ArgumentParser(description='Single Phase Safe DDPG')
parser.add_argument('--env_name', default="13bus", help='name of the environment to run')
parser.add_argument('--algorithm', default='safe-ddpg', help='name of algorithm')
parser.add_argument('--status', default='train')
parser.add_argument('--safe_type', default='three_single')  # loss, dd
args = parser.parse_args()
seed = 10
torch.manual_seed(seed)

# ========== 创建 DDPG 代理及相关参数 ==========
vlr = 2e-4  # value learning rate
plr = 1e-4  # policy learning rate
ph_num = 1  # 相数（1:单相）
max_ac = 0.3  # 最大动作幅度

# 根据选择的电网环境初始化参数
if args.env_name == '13bus':
    pp_net = create_13bus()
    injection_bus = np.array([2, 7, 9])
    env = IEEE13bus(pp_net, injection_bus)
    num_agent = len(injection_bus)

if args.env_name == '123bus':
    max_ac = 0.8
    pp_net = create_123bus()
    injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61]) - 1
    env = IEEE123bus(pp_net, injection_bus)
    num_agent = 14
    if args.algorithm == 'safe-ddpg':
        plr = 1.5e-4

if args.env_name == '13bus3p':
    injection_bus = np.array([633, 634, 671, 645, 646, 692, 675, 611, 652, 632, 680, 684])
    pp_net, injection_bus_dict = create_13bus3p(injection_bus)
    max_ac = 0.5
    env = IEEE13bus3p(pp_net, injection_bus_dict)
    num_agent = len(injection_bus)
    ph_num = 3
    if args.algorithm == 'safe-ddpg':
        plr = 1e-4
    if args.algorithm == 'ddpg':
        plr = 5e-5

# ========== 网络维度参数 ==========
obs_dim = env.obs_dim  # 观测空间维度
action_dim = env.action_dim  # 动作空间维度
hidden_dim = 100  # 隐藏层维度

if ph_num == 3:
    type_name = 'three-phase'
else:
    type_name = 'single-phase'

agent_list = []  # 初始化代理列表
replay_buffer_list = []  # 初始化回放缓冲区列表

# ========== 创建每个代理的策略网络、价值网络和缓冲区 ==========
for i in range(num_agent):
    if ph_num == 3:
        obs_dim = len(env.injection_bus[env.injection_bus_str[i]])
        action_dim = obs_dim
    value_net = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

    # 根据算法类型和相数选择相应的策略网络
    if args.algorithm == 'safe-ddpg' and not ph_num == 3:
        policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(
            device)
        target_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim,
                                              hidden_dim=hidden_dim).to(device)
    elif args.algorithm == 'safe-ddpg' and ph_num == 3 and args.safe_type == 'three_single':
        policy_net = SafePolicy3phase(env, obs_dim, action_dim, hidden_dim, env.injection_bus_str[i]).to(device)
        target_policy_net = SafePolicy3phase(env, obs_dim, action_dim, hidden_dim, env.injection_bus_str[i]).to(device)
    elif args.algorithm == 'linear':
        policy_net = LinearPolicy(env, ph_num)
        target_policy_net = LinearPolicy(env, ph_num)
    else:
        policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        target_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(
            device)

    target_value_net = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

    # 同步目标网络参数
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    # 创建 DDPG 代理并添加到代理列表
    agent = DDPG(policy_net=policy_net, value_net=value_net, target_policy_net=target_policy_net,
                 target_value_net=target_value_net, value_lr=vlr, policy_lr=plr)

    # 创建经验回放缓冲区
    replay_buffer = ReplayBufferPI(capacity=1000000)
    agent_list.append(agent)
    replay_buffer_list.append(replay_buffer)

# ========== 根据状态选择相应模式（训练或加载模型） ==========
if args.status == 'train':
    FLAG = 1
else:
    FLAG = 0


# 获取三相的索引
def get_id(phases):
    if phases == 'abc':
        id = [0, 1, 2]
    elif phases == 'ab':
        id = [0, 1]
    elif phases == 'ac':
        id = [0, 2]
    elif phases == 'bc':
        id = [1, 2]
    elif phases == 'a':
        id = [0]
    elif phases == 'b':
        id = [1]
    elif phases == 'c':
        id = [2]
    else:
        print("error!")
        exit(0)
    return id


# ========== 加载训练好的策略模型 ==========
if FLAG == 0:
    for i in range(num_agent):
        if ph_num == 3 and args.algorithm == 'safe-ddpg':
            policynet_dict = torch.load(
                f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/policy_net_checkpoint_a{i}.pth')
        else:
            policynet_dict = torch.load(
                f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/policy_net_checkpoint_a{i}.pth')
        agent_list[i].policy_net.load_state_dict(policynet_dict)

# ========== 训练模式 ==========
elif FLAG == 1:
    # 设置训练参数
    if args.algorithm == 'safe-ddpg':
        num_episodes = 200  # 对于 13bus3p
    else:
        num_episodes = 700  # 对于其他环境

    num_steps = 30  # 每个 episode 的步数
    if args.env_name == '123bus':
        num_steps = 60
    if args.env_name == 'eu-lv':
        num_steps = 30
    if args.env_name == '13bus3p':
        num_steps = 30
    batch_size = 256  # 批大小

    rewards = []
    avg_reward_list = []

    # 训练循环
    for episode in range(num_episodes):
        state = env.reset(seed=episode)
        episode_reward = 0
        last_action = np.zeros((num_agent, ph_num))

        for step in range(num_steps):
            action = []
            action_p = []
            for i in range(num_agent):
                # 根据当前策略和探索噪声采样动作
                if ph_num == 3:
                    action_agent = np.zeros(3)
                    phases = env.injection_bus[env.injection_bus_str[i]]
                    id = get_id(phases)
                    action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[i, id]])) + np.random.normal(0,
                                                                                                                    max_ac) / np.sqrt(
                        episode + 1)
                    action_tmp = action_tmp.reshape(len(id), )
                    for j in range(len(phases)):
                        action_agent[id[j]] = action_tmp[j]
                    action_agent = np.clip(action_agent, -max_ac, max_ac)
                    action_p.append(action_agent)
                else:
                    action_agent = agent_list[i].policy_net.get_action(np.asarray([state[i]])) + np.random.normal(0,
                                                                                                                  max_ac) / np.sqrt(
                        episode + 1)
                    action_agent = np.clip(action_agent, -max_ac, max_ac)
                    action_p.append(action_agent)
                action.append(action_agent)

            # PI 策略
            if ph_num == 3:
                action = last_action - np.asarray(action).reshape(-1, 3)
            else:
                action = last_action - np.asarray(action)

            # 执行动作并观察下一个状态和奖励
            next_state, reward, reward_sep, done = env.step_Preward(action, action_p)

            if (np.min(next_state) < 0.75):  # 如果电压超出范围，结束 episode
                break
            else:
                for i in range(num_agent):
                    if ph_num == 1:
                        state_buffer = state[i].reshape(ph_num, )
                        action_buffer = action[i].reshape(ph_num, )
                        last_action_buffer = last_action[i].reshape(ph_num, )
                        next_state_buffer = next_state[i].reshape(ph_num, )
                    else:
                        phases = env.injection_bus[env.injection_bus_str[i]]
                        id = get_id(phases)
                        state_buffer = state[i, id].reshape(len(phases), )
                        action_buffer = action[i, id].reshape(len(phases), )
                        last_action_buffer = last_action[i, id].reshape(len(phases), )
                        next_state_buffer = next_state[i, id].reshape(len(phases), )

                        # 存储 transition
                    replay_buffer_list[i].push(state_buffer, action_buffer, last_action_buffer, reward_sep[i],
                                               next_state_buffer, done)

                    # 更新网络
                    if ph_num == 3 and args.algorithm == 'safe-ddpg' and args.safe_type == 'loss':
                        if len(replay_buffer_list[i]) > batch_size:
                            agent_list[i].train_step_3ph(replay_buffer=replay_buffer_list[i], batch_size=batch_size)
                    else:
                        if len(replay_buffer_list[i]) > batch_size:
                            agent_list[i].train_step(replay_buffer=replay_buffer_list[i], batch_size=batch_size)

                if done:
                    episode_reward += reward
                    break
                else:
                    state = np.copy(next_state)
                    episode_reward += reward

            last_action = np.copy(action)

        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-40:])
        if (episode % 50 == 0):
            print(f"Episode * {episode} * Avg Reward is ==> {avg_reward}")
        avg_reward_list.append(avg_reward)

else:
    raise ValueError("Model loading option does not exist!")

# ========== 训练结束，生成奖励图像 ==========
if args.status == 'train':
    check_buffer = replay_buffer_list[0]
    buffer_len = replay_buffer_list[0].__len__()
    state, action, last_action, reward, next_state, done = replay_buffer_list[0].sample(buffer_len - 1)
    if ph_num == 1:
        plt.scatter(action, reward)
        plt.title('bus 0')
        plt.savefig('bus0.png')
        plt.show()
    else:
        plt.scatter(action[:, 0], reward)
        plt.title('bus 0')
        plt.savefig('bus0.png')
        plt.show()

# ========== 测试代理策略 ==========
state = env.reset()
episode_reward = 0
last_action = np.zeros((num_agent, 1))
action_list = []
state_list = []
reward_list = []
state_list.append(state)

for step in range(100):
    action = []
    for i in range(num_agent):
        if ph_num == 3:
            action_agent = np.zeros(3)
            phases = env.injection_bus[env.injection_bus_str[i]]
            id = get_id(phases)
            action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[i, id]]))
            action_tmp = action_tmp.reshape(len(id), )
            for i in range(len(phases)):
                action_agent[id[i]] = action_tmp[i]
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)
        else:
            action_agent = agent_list[i].policy_net.get_action(np.asarray([state[i]]))
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

    # PI 策略
    action = last_action - np.asarray(action)

    # 执行动作并观察奖励和下一个状态
    next_state, reward, reward_sep, done = env.step_Preward(action, (last_action - action))
    reward_list.append(reward)
    if done:
        print("finished")
    action_list.append(last_action - action)
    state_list.append(next_state)
    last_action = np.copy(action)
    state = next_state

# ========== 绘制状态和动作变化图 ==========
fig, axs = plt.subplots(1, num_agent + 1, figsize=(15, 3))
for i in range(num_agent):
    axs[i].plot(range(len(action_list)), np.array(state_list)[:len(action_list), i], '-.', label='states')
    axs[i].legend(loc='lower left')

fig1, axs1 = plt.subplots(1, num_agent + 1, figsize=(15, 3))
for i in range(num_agent):
    axs1[i].plot(range(len(action_list)), np.array(action_list)[:len(action_list), i], '-.', label='actions')
    axs1[i].legend(loc='lower left')

axs[num_agent].plot(range(len(reward_list)), reward_list)
plt.show()
