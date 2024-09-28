from ieee123bus_env import IEEE123bus
from MADDPG import MADDPG
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandapower.converter as pc

def create_123bus(pv_buses, es_buses):
    pp_net = pc.from_mpc('pandapower models/pandapower models/case_123.mat', casename_mpc_file='case_mpc')

    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    for bus in pv_buses:
        if bus in pp_net.bus.index:
            pp.create_sgen(pp_net, bus, p_mw=1.0, q_mvar=0.0)
        else:
            print(f"  Warning: Bus {bus} not found in network bus index")

    for bus in es_buses:
        if bus in pp_net.bus.index:
            pp.create_storage(pp_net, bus=bus, p_mw=0.5, max_e_mwh=2.0, soc_percent=50, min_e_mwh=0, q_mvar=0.1)
        else:
            print(f"  Warning: Bus {bus} not found in network bus index")

    return pp_net


# 定义 PV 和 ES 节点
pv_buses = np.array([9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60])
es_buses = np.array([20, 30, 40])

# 创建 Pandapower 网络对象
pp_net = create_123bus(pv_buses, es_buses)

# 定义智能体参数
pv_params = (5, 1, 64)  # 观测维度，动作维度，隐藏层维度
storage_params = (5, 2, 64)  # 观测维度，动作维度，隐藏层维度

# 创建 MADDPG 实例
maddpg = MADDPG(pv_params, storage_params, pv_buses, es_buses, gamma=0.99, tau=0.01, buffer_size=100000, batch_size=64)

# # 训练模型并记录电压数据
# voltage_data, over_limit_rates = maddpg.train(num_episodes=500, pp_net=pp_net, pv_bus=pv_buses, es_bus=es_buses)
# # 保存模型
# maddpg.save_model('model_directory')
#
# # 绘制电压越线情况
# voltage_data = np.array(voltage_data)
# num_steps = voltage_data.shape[0]
# num_buses = voltage_data.shape[1]

# 加载预训练模型
maddpg.load_model('model_directory')

# 在线训练模型并记录电压数据
voltage_data, over_limit_rates = maddpg.online_train(num_steps=800, pp_net=pp_net, pv_bus=pv_buses, es_bus=es_buses)


# 绘制电压越限率随时间的变化
plt.figure(figsize=(10, 5))
plt.plot(over_limit_rates, label='Voltage Limit Exceedance Rate')
plt.xlabel('Step')
plt.ylabel('Over Limit Rate')
plt.title('Voltage Limit Exceedance Rate Over Time')
plt.legend()
plt.show()
