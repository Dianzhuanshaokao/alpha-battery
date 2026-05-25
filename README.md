# AlphaBattery

本仓库现在按功能拆成四个顶层板块，便于区分需求、模型、仿真和强化学习优化：

- `01_requirements/`: 项目需求、研究范围、数据契约、主线说明
- `02_pybamm_model/`: PyBaMM 相关模型代码、案例配置、模型工具
- `03_simulation/`: 可执行仿真入口、notebook、输入数据与结果
- `04_rl_optimization/`: `SmartCharging` 强化学习与优化 demo
- `external/pybamm/`: vendored PyBaMM 上游依赖

每个板块现在都统一提供：

- `main.py`
- `README.md`
- `outputs/`

## Quick Navigation

- 闭环退化建模需求：[01_requirements/alpha_closed_loop.md](/Z:/AlphaBattery/01_requirements/alpha_closed_loop.md)
- 论文复现需求：[01_requirements/reproduce_li2024.md](/Z:/AlphaBattery/01_requirements/reproduce_li2024.md)
- PyBaMM 模型板块：[02_pybamm_model/README.md](/Z:/AlphaBattery/02_pybamm_model/README.md)
- 仿真板块：[03_simulation/README.md](/Z:/AlphaBattery/03_simulation/README.md)
- RL 优化板块：[04_rl_optimization/README.md](/Z:/AlphaBattery/04_rl_optimization/README.md)
- 集群说明：[cluster_usage.md](/Z:/AlphaBattery/01_requirements/cluster_usage.md)

## Main Entrypoints

需求与集群导航：

```bash
python 01_requirements/main.py
```

模型板块入口：

```bash
python 02_pybamm_model/main.py
```

仿真板块入口：

```bash
python 03_simulation/main.py
```

RL 训练：

```bash
python 04_rl_optimization/main.py
```

## Notes

- `pybamm/` 已移到 `external/pybamm/`，不再与本地项目代码混放
- 结果文件仍保留在仓库中，但已跟随各自板块归位
- `SmartCharging` 的 Bayesian optimization 入口已归位，但仍缺少专用 `config.yaml`，当前不视为已打通
- 仿真结果可先导出到 `03_simulation/outputs/tensorboard/`，再用 TensorBoard 查看
- 强化学习训练结果默认写到 `04_rl_optimization/outputs/training/`
