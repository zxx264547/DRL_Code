import numpy as np
np.Inf = np.inf
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math
import MADDPG
# 定义一个123总线的类，
class IEEE123bus(gym.Env):
    def __init__(self, pp_net, pv_bus, es_bus, v0=1, vmax=1.05, vmin=0.95):
        self.network = pp_net  # pp_net 是通过 create_123bus() 函数创建的 Pandapower 网络对象。
        self.obs_dim = 5  # 观测维度
        self.pv_action_dim = 1  # 动作维度
        self.es_action_dim = 2  # 动作维度
        self.injection_bus = pv_bus + es_bus  # 电力注入的节点（可以控制的节点）
        self.agentnum = len(self.injection_bus)
        self.pv_agentnum = len(pv_bus)
        self.es_agentnum = len(es_bus)

        self.v0 = v0  # 初始电压值
        self.vmax = vmax 
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])  # 初始负载的有功功率
        self.load0_q = np.copy(self.network.load['q_mvar'])  # 初始负载的无功功率

        self.gen0_p = np.copy(self.network.sgen['p_mw'])  # 初始发电机的有功功率
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])  # 初始发电机的无功功率

        self.storage0_p = np.copy(self.network.storage['p_mw'])
        self.storage0_p = np.copy(self.network.storage['max_e_mwh'])
        self.storage0_p = np.copy(self.network.storage['soc_percent'])
        self.storage0_p = np.copy(self.network.storage['min_e_mwh'])
        self.storage0_p = np.copy(self.network.storage['q_mvar'])
        # todo:状态是什么
        self.state = np.ones(self.agentnum, )  # 每个智能体节点的初始状态为1
#     功能：给定动作的情况下，计算新的状态
#     输入：
    #     action：一个数组，表示在当前步中采取的动作
    #     p_action：一个数组，表示前一个动作。
#     输出：
#
#

    def step_Preward(self, state, actions, next_state, V_target=1.0, alpha=1.0, beta=1.0, gamma=1.0):
        # todo:reward
        """
        计算奖励函数

        参数:
        state: 当前状态（电压、功率等）
        actions: 智能体的动作（调整后的无功功率、有功功率等）
        next_state: 下一状态（电压、功率等）
        V_target: 目标电压值（标幺值）
        alpha: 电压稳定性权重
        beta: 能量平衡权重
        gamma: 合作奖励权重

        返回:
        total_reward: 总奖励值
        """
        # 假设状态包含电压和功率信息
        V = next_state[:, 0]  # 电压
        P_load = next_state[:, 1]  # 负载有功功率
        P_pv = next_state[:, 2]  # 光伏有功功率
        P_es = next_state[:, 3]  # 储能有功功率
        Q_load = next_state[:, 4]  # 负载无功功率
        Q_pv = next_state[:, 5]  # 光伏无功功率
        Q_es = next_state[:, 6]  # 储能无功功率

        # 电压稳定性奖励
        voltage_deviation = np.abs(V - V_target)
        voltage_reward = -alpha * np.sum(voltage_deviation ** 2)

        # 能量平衡奖励
        P_balance = P_load - P_pv - P_es
        Q_balance = Q_load - Q_pv - Q_es
        energy_balance_reward = -beta * (np.sum(P_balance ** 2) + np.sum(Q_balance ** 2))

        # 合作奖励（示例：智能体间的通信或全局目标）
        cooperation_reward = gamma * np.sum(np.dot(actions, actions.T))

        # 总奖励
        total_reward = voltage_reward + energy_balance_reward + cooperation_reward

        return total_reward


        
        # # state-transition dynamics
        # # 根据动作更新网络中的无功功率
        # for i in range(len(self.injection_bus)):
        #     self.network.sgen.at[i, 'q_mvar'] = action[i]
        # # 运行潮流计算runpp
        # pp.runpp(self.network, algorithm='bfsw', init='dc')
        # # 更新网络状态
        # self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        # # 如果所有注入节点的电压状态在 [0.95, 1.05) 范围内，标记任务完成。
        # if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
        #     done = True
        # # 返回新的状态、奖励、局部奖励和 done 标志。
        # return self.state, reward, reward_sep, done

    # step_load方法在给定动作（action）、负载有功功率（load_p）和无功功率（load_q）的情况下，
    # 计算新的状态、奖励以及是否完成任务（done标志）。
    def step_load(self, action, load_p, load_q): #state-transition with specific load
        
        done = False 
        
        reward = float(-50*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)
        
        #adjust power consumption at the load bus
        # 遍历所有的智能体节点，把智能体消耗的有功和无功调整到给定的值
        for i in range(self.env.agentnum):
            self.network.load.at[i, 'p_mw'] = load_p[i]
            self.network.load.at[i, 'q_mvar'] = load_q[i]
           
        #adjust reactive power inj at the PV bus
        # 调整无功节点的无功功率注入
        for i in range(self.env.agentnum):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 
        # 潮流计算
        pp.runpp(self.network, algorithm='bfsw', init='dc')
        # 更新状态
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, state_all, reward, done
    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        # 随机选择一个场景，0代表低电压场景，1代表高电压场景
        senario = np.random.choice([0,1])
        # senario=0，低电压场景
        # 在低电压情景中，将所有发电机（sgen）和负载（load）的初始功率设置为 0。
        # 然后为每个分布式电源（sgen）随机分配一个负功率（即消耗功率），以模拟低电压情况。
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = -0.8*np.random.uniform(15, 60)
            # self.network.sgen.at[0, 'q_mvar'] = -0.8*np.random.uniform(10, 300)
            self.network.sgen.at[1, 'p_mw'] = -0.8*np.random.uniform(10, 45)
            self.network.sgen.at[2, 'p_mw'] = -0.8*np.random.uniform(10, 55)
            self.network.sgen.at[3, 'p_mw'] = -0.8*np.random.uniform(10, 30)
            self.network.sgen.at[4, 'p_mw'] = -0.6*np.random.uniform(1, 35)
            self.network.sgen.at[5, 'p_mw'] = -0.5*np.random.uniform(2, 25)
            self.network.sgen.at[6, 'p_mw'] = -0.8*np.random.uniform(2, 30)
            self.network.sgen.at[7, 'p_mw'] = -0.9*np.random.uniform(1, 10)
            self.network.sgen.at[8, 'p_mw'] = -0.7*np.random.uniform(1, 15)
            self.network.sgen.at[9, 'p_mw'] = -0.5*np.random.uniform(1, 30)
            self.network.sgen.at[10, 'p_mw'] = -0.3*np.random.uniform(1, 20)
            self.network.sgen.at[11, 'p_mw'] = -0.5*np.random.uniform(1, 20)
            self.network.sgen.at[12, 'p_mw'] = -0.4*np.random.uniform(1, 20)
            self.network.sgen.at[13, 'p_mw'] = -0.4*np.random.uniform(2, 10)
            #not real controllers
            self.network.sgen.at[14, 'p_mw'] = -0.4*np.random.uniform(10, 20)
            self.network.sgen.at[15, 'p_mw'] = -0.8*np.random.uniform(10, 20)
            self.network.sgen.at[16, 'p_mw'] = -0.8*np.random.uniform(10, 20)

        # 将所有发电机和负载的初始功率设置为 0。
        # 为每个发电机随机分配一个正的有功功率，以模拟高电压情况。
        elif(senario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = 0.8*np.random.uniform(15, 60)
            # self.network.sgen.at[0, 'q_mvar'] = 0.6*np.random.uniform(5, 300)
            self.network.sgen.at[1, 'p_mw'] = 0.8*np.random.uniform(15, 50)
            self.network.sgen.at[2, 'p_mw'] = 0.8*np.random.uniform(20, 60)
            self.network.sgen.at[3, 'p_mw'] = 0.8*np.random.uniform(10, 34)
            self.network.sgen.at[4, 'p_mw'] = 0.8*np.random.uniform(2, 20)
            self.network.sgen.at[5, 'p_mw'] = 0.8*np.random.uniform(2, 80)
            self.network.sgen.at[6, 'p_mw'] = 0.8*np.random.uniform(10, 80)
            self.network.sgen.at[7, 'p_mw'] = 0.8*np.random.uniform(5, 50)
            self.network.sgen.at[8, 'p_mw'] = 0.7*np.random.uniform(2, 30)
            self.network.sgen.at[9, 'p_mw'] = 0.5*np.random.uniform(2, 30)
            self.network.sgen.at[10, 'p_mw'] = 0.4*np.random.uniform(1, 40)
            self.network.sgen.at[11, 'p_mw'] = 0.5*np.random.uniform(1, 30)
            self.network.sgen.at[12, 'p_mw'] = 0.5*np.random.uniform(1, 30)
            self.network.sgen.at[13, 'p_mw'] = 0.5*np.random.uniform(1, 24)
            #not real controllers
            self.network.sgen.at[14, 'p_mw'] = 0.5*np.random.uniform(15, 25)
            self.network.sgen.at[15, 'p_mw'] = 0.8*np.random.uniform(10, 50)
            self.network.sgen.at[16, 'p_mw'] = 0.8*np.random.uniform(10, 20)
            
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state
    
    def reset0(self, seed=1): #reset voltage to nominal value
        
        self.network.load['p_mw'] = 0*self.load0_p
        self.network.load['q_mvar'] = 0*self.load0_q

        self.network.sgen['p_mw'] = 0*self.gen0_p
        self.network.sgen['q_mvar'] = 0*self.gen0_q
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

def create_123bus():
    pp_net = pp.converter.from_mpc('pandapower models/pandapower models/case_123.mat', casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 9, p_mw = 1.5, q_mvar=0)
    pp.create_sgen(pp_net, 10, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 15, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 19, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 32, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 35, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 47, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 58, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 65, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 74, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 82, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 91, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 103, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 60, p_mw = 1, q_mvar=0) #node 114 in the png



    # 添加储能系统
    pp.create_storage(pp_net, bus=20, p_mw=0.5, max_e_mwh=2.0, soc_percent=50, min_e_mwh=0, q_mvar=0.1)
    pp.create_storage(pp_net, bus=30, p_mw=0.8, max_e_mwh=3.0, soc_percent=50, min_e_mwh=0, q_mvar=0.2)
    pp.create_storage(pp_net, bus=40, p_mw=0.6, max_e_mwh=2.5, soc_percent=50, min_e_mwh=0, q_mvar=0.15)

    #only for reset
    pp.create_sgen(pp_net, 13, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 14, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 18, p_mw = 1, q_mvar=0)

    return pp_net


if __name__ == "__main__":
    net = create_123bus()
    pv_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61]) - 1
    es_bus = np.array([20, 30, 40])
    env = IEEE123bus(net, pv_bus, es_bus)

    # 多智能体设置
    pv_num_agents = env.pv_agentnum
    es_num_agents = env.es_agentnum
    state_dim = 1
    pv_action_dim = 1
    es_action_dim = 2
    hidden_dim = 128  # 隐藏层维度

    # 创建多智能体 DDPG 智能体
    agents = []
    for _ in range(pv_num_agents):
        policy_net = MADDPG.PolicyNetwork(state_dim, pv_action_dim, hidden_dim)
        value_net = MADDPG.ValueNetwork(state_dim, pv_action_dim, hidden_dim)
        target_policy_net = MADDPG.PolicyNetwork(env, state_dim, pv_action_dim, hidden_dim)
        target_value_net = MADDPG.ValueNetwork(state_dim, action_dim, hidden_dim)
        agent = MADDPG.DDPG(policy_net, value_net, target_policy_net, target_value_net)
        agents.append(agent)

    # 创建经验回放缓冲区
    replay_buffer = MADDPG.ReplayBuffer(capacity=10000)

    num_episodes = 1000
    max_steps = 200
    batch_size = 64

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            actions = np.array([agent.policy_net.get_action(state[i:i+1]) for i, agent in enumerate(agents)])
            next_state, reward, done = env.step_load(actions, env.network.load['p_mw'], env.network.load['q_mvar'])
            last_action = actions.copy()
            replay_buffer.push(state, actions, last_action, reward, next_state, done)

            if len(replay_buffer) > batch_size:
                for agent in agents:
                    agent.train_step(replay_buffer, batch_size)

            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")

    # 绘制电压越限率和有功功率损失
    num_steps = 200
    voltage_limits = (0.95, 1.05)
    voltage_violation_rate = np.zeros(num_steps)
    active_power_loss = np.zeros(num_steps)

    for i in range(num_steps):
        state = env.reset(i)
        voltage_violation_rate[i] = np.mean((state < voltage_limits[0]) | (state > voltage_limits[1]))
        active_power_loss[i] = np.sum(env.network.res_line.pl_mw)

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(voltage_violation_rate, label='Voltage Violation Rate')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Voltage Violation Rate')
    ax1.set_title('Voltage Violation Rate Over Time')
    ax1.legend()
    plt.show()

    fig, ax2 = plt.subplots(figsize=(15, 5))
    ax2.plot(active_power_loss, label='Active Power Loss', color='orange')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Active Power Loss (MW)')
    ax2.set_title('Active Power Loss Over Time')
    ax2.legend()
    plt.show()


