# Reproduce_Li2024 Requirements

## Scope

`Reproduce_Li2024` 这条线是一个纯 `PyBaMM` 退化仿真复现项目，不依赖真实实验数据对比。

目标是：

- 给定一个案例 JSON
- 自动执行 ageing block
- 按累计放电量定义等效循环
- 在检查点执行 `LongRPT` 或 `ShortRPT`
- 从 RPT 中提取 `SOH / LLI / LAM / 内阻`

## Current Layout

- 需求说明：`01_requirements/reproduce_li2024.md`
- 模型核心：`02_pybamm_model/reproduce_li2024_model/`
- 运行入口、输出、notebook：`03_simulation/reproduce_li2024/`
- 上游 PyBaMM 依赖：`external/pybamm/`

## Model Assets

- 模型代码：`02_pybamm_model/reproduce_li2024_model/sim/`
- 案例配置：`02_pybamm_model/reproduce_li2024_model/configs/cases/`

## Simulation Assets

- CLI 入口：`03_simulation/reproduce_li2024/scripts/run_from_json.py`
- 可视化：`03_simulation/reproduce_li2024/notebooks/visualize_simulation.ipynb`
- 当前输出：`03_simulation/reproduce_li2024/outputs/`
- 历史材料：`03_simulation/reproduce_li2024/legacy/`
- 集群脚本：`03_simulation/reproduce_li2024/slurm/run_reproduce_li2024.sbatch`

## Main Entrypoint

```bash
python 03_simulation/reproduce_li2024/scripts/run_from_json.py \
  --config 02_pybamm_model/reproduce_li2024_model/configs/cases/okane2023_full_cycle.json \
  --output-root 03_simulation/reproduce_li2024/outputs
```

默认情况下，脚本会优先查找仓库内的 `external/pybamm/`；也可以继续用 `--pybamm-root` 或环境变量显式指定外部 PyBaMM 路径。

## Current Notes

- `legacy/Fun_NC.py` 及旧 notebook 仅作历史参考
- 新结构里，模型定义和仿真资产已分拆到第二、第三板块
