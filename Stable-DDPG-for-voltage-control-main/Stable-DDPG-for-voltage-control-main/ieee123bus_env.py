import gym
from gym import spaces
import numpy as np
import pandapower as pp


class IEEE123bus(gym.Env):
    def __init__(self, pp_net, pv_bus, es_bus, v0=1, vmax=1.05, vmin=0.95):
        super(IEEE123bus, self).__init__()

        self.network = pp_net  # pp_net 是通过 create_123bus() 函数创建的 Pandapower 网络对象。
        self.obs_dim = 5  # 观测维度
        self.pv_action_dim = 1  # 光伏动作维度
        self.es_action_dim = 2  # 储能动作维度
        self.pv_buses = list(pv_bus)  # 光伏智能体控制的节点
        self.es_buses = list(es_bus)  # 储能智能体控制的节点
        # 存储光伏和储能节点的索引
        self.pv_buses_index = list(range(0, len(pv_bus)))
        self.es_buses_index = list(range(0, len(es_bus)))

        self.v0 = v0  # 初始电压值
        self.vmax = vmax
        self.vmin = vmin

        # 定义光伏智能体的状态空间和动作空间
        self.pv_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.pv_action_space = spaces.Box(low=-1, high=1, shape=(self.pv_action_dim,), dtype=np.float32)

        # 定义储能智能体的状态空间和动作空间
        self.storage_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.storage_action_space = spaces.Box(low=-1, high=1, shape=(self.es_action_dim,), dtype=np.float32)

        # 初始化环境状态
        self.state_pv = self._get_initial_state('pv', self.pv_buses)
        self.state_storage = self._get_initial_state('storage', self.es_buses)

    def reset_pv(self):
        # 重置光伏智能体的状态
        self.state_pv = self._get_initial_state('pv', self.pv_buses)  # 基于潮流计算得到初始状态
        return self.state_pv

    def reset_storage(self):
        # 重置储能智能体的状态
        self.state_storage = self._get_initial_state('storage', self.es_buses)  # 基于潮流计算得到初始状态
        return self.state_storage

    def step_pv(self, actions):
        next_states = []
        rewards = []
        dones = []
        for i, bus in enumerate(self.pv_buses):
            self._apply_action(actions[i], 'pv', bus)
        try:
            pp.runpp(self.network, max_iteration=50)
        except pp.powerflow.LoadflowNotConverged:
            print("Power flow for PV did not converge")
            return next_states, rewards, [True] * len(self.pv_buses)  # 终止当前回合
        for i, bus in enumerate(self.pv_buses):
            next_state = self._get_state('pv', bus)
            reward = self._calculate_reward(next_state, 'pv')
            done = self._check_done(next_state, 'pv')
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return next_states, rewards, dones

    def step_storage(self, actions):
        next_states = []
        rewards = []
        dones = []
        for i, bus in enumerate(self.es_buses):
            self._apply_action(actions[i], 'storage', bus)
        try:
            pp.runpp(self.network, max_iteration=50)
        except pp.powerflow.LoadflowNotConverged:
            print("Power flow for storage did not converge")
            return next_states, rewards, [True] * len(self.es_buses)  # 终止当前回合
        for i, bus in enumerate(self.es_buses):
            next_state = self._get_state('storage', bus)
            reward = self._calculate_reward(next_state, 'storage')
            done = self._check_done(next_state, 'storage')
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return next_states, rewards, dones

    def _get_initial_state(self, agent_type, buses):
        # 获取初始状态的逻辑
        pp.runpp(self.network)
        return [self._get_state(agent_type, bus) for bus in buses]

    def _get_state(self, agent_type, bus):
        if agent_type == 'pv':
            if bus in self.network.sgen['bus'].values:
                sgen_idx = self.network.sgen[self.network.sgen['bus'] == bus].index[0]
                p_mw = self.network.sgen.at[sgen_idx, 'p_mw']
                q_mvar = self.network.sgen.at[sgen_idx, 'q_mvar']
                v_pu = self.network.res_bus.at[bus, 'vm_pu']
                load_p_mw = self.network.load.at[bus, 'p_mw'] if bus in self.network.load['bus'].values else 0.0
                load_q_mvar = self.network.load.at[bus, 'q_mvar'] if bus in self.network.load['bus'].values else 0.0
                return np.array([p_mw, q_mvar, load_p_mw, load_q_mvar, v_pu])
            else:
                print(f"Warning: Bus {bus} not found in sgen")
                return None
        elif agent_type == 'storage':
            if bus in self.network.storage['bus'].values:
                storage_idx = self.network.storage[self.network.storage['bus'] == bus].index[0]
                p_mw = self.network.storage.at[storage_idx, 'p_mw']
                q_mvar = self.network.storage.at[storage_idx, 'q_mvar']
                v_pu = self.network.res_bus.at[bus, 'vm_pu']
                load_p_mw = self.network.load.at[bus, 'p_mw'] if bus in self.network.load['bus'].values else 0.0
                load_q_mvar = self.network.load.at[bus, 'q_mvar'] if bus in self.network.load['bus'].values else 0.0
                return np.array([p_mw, q_mvar, load_p_mw, load_q_mvar, v_pu])
            else:
                print(f"Warning: Bus {bus} not found in storage")
                return None

    def _apply_action(self, action, agent_type, bus):
        if agent_type == 'pv':
            if bus in self.network.sgen['bus'].values:
                sgen_idx = self.network.sgen[self.network.sgen['bus'] == bus].index[0]
                # 确保动作在合理范围内
                # q_mvar = np.clip(float(action), -1.0, 1.0)
                # self.network.sgen.loc[sgen_idx, 'q_mvar'] = q_mvar
                self.network.sgen.loc[sgen_idx, 'q_mvar'] = float(action)
            else:
                print(f"Warning: Bus {bus} not found in sgen")
        elif agent_type == 'storage':
            if bus in self.network.storage['bus'].values:
                storage_idx = self.network.storage[self.network.storage['bus'] == bus].index[0]
                # 确保动作在合理范围内
                # p_mw = np.clip(float(action[0]), -1.0, 1.0)
                # q_mvar = np.clip(float(action[1]), -1.0, 1.0)
                # self.network.storage.loc[storage_idx, 'p_mw'] = p_mw
                # self.network.storage.loc[storage_idx, 'q_mvar'] = q_mvar
                self.network.storage.loc[storage_idx, 'p_mw'] = float(action[0])
                self.network.storage.loc[storage_idx, 'q_mvar'] = float(action[1])
            else:
                print(f"Warning: Bus {bus} not found in storage")

    def _calculate_reward(self, state, agent_type):
        voltage = state[-1]
        reward = -abs(voltage - 1.0)  # 以电压偏离1.0的绝对值为负奖励
        if voltage < self.vmin or voltage > self.vmax:
            reward -= 1.0  # 如果电压超出范围，给予额外的负奖励
        return reward

    def _check_done(self, state, agent_type):
        voltage = state[-1]
        return voltage < self.vmin or voltage > self.vmax

