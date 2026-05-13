# 浓密脱水场景说明

本文档只记录当前 TSF 仓库采用的固定浓密脱水协议。场景说明用于理解训练/测试划分和过程扰动，不作为额外输入特征，也不生成场景专属变量 embedding。

## 任务口径

- 任务：分钟级底流浓度 `underflow_concentration` 软测量。
- 可见输入：`q_in`, `p2`, `p3`, `phase_pressurizing`, `phase_discharging`。
- 标签：当前分钟 `underflow_concentration`。
- 窗口：30 分钟历史窗口，包含当前目标分钟。
- 环境：面向软测量评测的最小动力学环境，不是真实工业原始导出。

## 训练与近域

| 场景 | 分组 | 含义 |
| --- | --- | --- |
| `train_jitter_1` 到 `train_jitter_6` | `train` | 标称工况家族内的轻微分段来料流量波动 |
| `near_train_like_jitter` | `near` | 与训练同家族但扰动数值不同，用于近域泛化测试 |

训练组和近域组不引入隐藏物理变化或观测失真，主要改变 `q_in` 轨迹。

## 远域测试

| 场景 | 类型 | 当前输入下是否直接可见 | 简要含义 |
| --- | --- | --- | --- |
| `far_constant_high_load` | 真实过程变化 | 是 | 上游持续高负荷来料，`q_in` 长期偏高 |
| `far_constant_low_load` | 真实过程变化 | 是 | 上游持续低负荷来料，`q_in` 长期偏低 |
| `far_discharge_efficiency_drop` | 真实过程变化 | 否 | 机械排料或执行链效率下降，卸压/排料段变慢 |
| `far_slurry_property_shift` | 真实过程变化 | 否 | 矿浆性质变化，使压力到浓度映射漂移 |
| `far_pressure_sensor_bias` | 观测变化 | 是，但观测有偏 | 底压传感器标定偏差，模型看到的 `p3` 偏高 |
| `far_q_in_scale_error` | 观测变化 | 是，但观测有偏 | 流量计缩放误差，模型看到的 `q_in` 偏大 |
| `far_feed_solids_ratio_shift` | 隐藏真实过程变化 | 否 | 相近体积流量下有效固体负荷不同 |
| `far_flocculant_dosing_failure` | 隐藏真实过程变化 | 否 | 絮凝剂不足或失效，沉降/压实效果下降 |
| `far_media_permeability_loss` | 隐藏真实过程变化 | 否 | 脱水介质堵塞或渗透性下降，排出变慢 |
| `far_actuator_deadtime_gain_loss` | 隐藏真实过程变化 | 否 | 执行器响应迟滞或升压增益下降 |
| `far_pressure_signal_delay` | 观测变化 | 是，但观测滞后 | 底压信号存在通讯或刷新延迟 |
| `far_restart_inventory_shift` | 隐藏真实过程变化 | 否 | 启停后库存、床层结构或残余压实状态改变 |

## 实现边界

- `q_in` 表示进入浓密/脱水系统的浆体体积流量。
- `phase_pressurizing` 和 `phase_discharging` 是操作阶段 one-hot 特征。
- 隐藏过程变化默认只改变真实 plant 或观测生成，不新增显式输入列。
- 公开 `windows.npz` 的 `sample_metadata_json` 保存每个样本的分组和场景字段；训练脚本以该 NumPy bundle 为准读取样本。
