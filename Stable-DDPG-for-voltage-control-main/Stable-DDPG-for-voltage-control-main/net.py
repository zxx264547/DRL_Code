import pandapower as pp
import pandapower.converter as pc
import scipy.io
import matplotlib.pyplot as plt

# # 从MAT文件加载数据
# mat_data = scipy.io.loadmat('network.mat')
#
# # 将MAT数据转换为Pandapower网络
# net = pc.from_mpc(mat_data, casename_mpc_file='network', f_hz=50.0, casename_py_file=None)
net = pp.converter.from_mpc('pandapower models/pandapower models/case_123.mat', casename_mpc_file='case_mpc')
net.sgen['p_mw'] = 0.0
net.sgen['q_mvar'] = 0.0

pp.create_sgen(net, 9, p_mw=1.5, q_mvar=0)
pp.create_sgen(net, 10, p_mw=1, q_mvar=0)
pp.create_sgen(net, 15, p_mw=1, q_mvar=0)
pp.create_sgen(net, 19, p_mw=1, q_mvar=0)
pp.create_sgen(net, 32, p_mw=1, q_mvar=0)
pp.create_sgen(net, 35, p_mw=1, q_mvar=0)
pp.create_sgen(net, 47, p_mw=1, q_mvar=0)
pp.create_sgen(net, 58, p_mw=1, q_mvar=0)
pp.create_sgen(net, 65, p_mw=1, q_mvar=0)
pp.create_sgen(net, 74, p_mw=1, q_mvar=0)
pp.create_sgen(net, 82, p_mw=1, q_mvar=0)
pp.create_sgen(net, 91, p_mw=1, q_mvar=0)
pp.create_sgen(net, 103, p_mw=1, q_mvar=0)
pp.create_sgen(net, 60, p_mw=10, q_mvar=0)

# 添加储能系统
pp.create_storage(net, bus=20, p_mw=0.5, max_e_mwh=2.0, soc_percent=50, min_e_mwh=0, q_mvar=0.1)
pp.create_storage(net, bus=30, p_mw=0.8, max_e_mwh=3.0, soc_percent=50, min_e_mwh=0, q_mvar=0.2)
pp.create_storage(net, bus=40, p_mw=0.6, max_e_mwh=2.5, soc_percent=50, min_e_mwh=0, q_mvar=0.15)

# 运行潮流计算
pp.runpp(net)

# 获取初始电压分布
voltages = net.res_bus.vm_pu

# 打印初始电压
print("Initial Voltages (pu):", voltages)

# 可视化电压分布
plt.figure(figsize=(10, 6))
plt.plot(voltages, marker='o')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (pu)')
plt.title('Initial Voltage Profile')
plt.grid()
plt.show()
