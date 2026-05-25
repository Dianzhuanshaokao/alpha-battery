# 04 RL Optimization

这个板块放 `SmartCharging` 强化学习与优化 demo。

统一骨架：

- 入口脚本：[main.py](/Z:/AlphaBattery/04_rl_optimization/main.py)
- 输出目录：`04_rl_optimization/outputs/`

## Contents

- `SmartCharging/Scripts/RLtrain/`: PPO 训练入口与环境
- `SmartCharging/Scripts/BatesianOptimization/`: Bayesian optimization 相关脚本
- `SmartCharging/Scripts/CustomizedPlotting/`: 绘图与日志解析
- `SmartCharging/Datas/`: 训练日志、CSV、历史结果

## Main Entrypoint

```bash
python 04_rl_optimization/main.py train
```

## Current Status

- RL 训练入口已按新目录修正本地导入与默认输出位置
- `main_bo.py` 已迁入新结构，但仍依赖缺失的 `config.yaml`
- 因此本板块当前只有 RL 训练入口可视为已对齐新结构

## TensorBoard

新训练结果默认写入 `04_rl_optimization/outputs/training/`。

推荐命令：

```bash
tensorboard --logdir 04_rl_optimization/outputs/training
```

## Cluster

- 远程训练默认在 `121.48.164.50`
- 使用 GPU 前先执行 `module load gpu`
- GPU 节点：`c1`、`c2`、`c3`
- 其他节点默认视为 CPUCompute

更完整的集群使用说明见：

- [cluster_usage.md](/Z:/AlphaBattery/01_requirements/cluster_usage.md)
