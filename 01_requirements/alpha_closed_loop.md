# AlphaBattery Closed-Loop Requirements

## Scope

这条主线聚焦于“真实数据 + PyBaMM 闭环退化建模”：

1. 读取真实循环协议与真实 RPT 数据
2. 用相同协议驱动 PyBaMM 耦合退化模型
3. 从仿真 RPT 提取 `SOH / DCIR / HPPC / LLI / LAM / dQ/dV / ICA`
4. 计算真实 RPT 与仿真 RPT 的误差
5. 通过优化更新超参数
6. 输出状态向量与最新仿真产物

## Current Layout

- 需求说明：`01_requirements/`
- 模型资产：`02_pybamm_model/closed_loop_model/`
- 仿真入口与结果：`03_simulation/closed_loop/`
- 上游 PyBaMM 代码：`external/pybamm/`

## Core Model Assumptions

- SEI 生长：`SEI = solvent-diffusion limited`
- 析锂：`lithium plating = partially reversible`
- LAM：`loss of active material = stress-driven`
- 颗粒力学：`particle mechanics = swelling and cracking`
- 电解液干涸：通过外部 dryout profile 输入孔隙率缩放

## Main Inputs

- `03_simulation/closed_loop/data/cycle_protocols.csv`
- `03_simulation/closed_loop/data/real_cycle_timeseries.csv`
- `03_simulation/closed_loop/data/real_rpt_summary.csv`
- `03_simulation/closed_loop/data/real_rpt_low_rate_trace.csv`
- `03_simulation/closed_loop/data/electrolyte_dryout_profile.csv`

## Main Model Configs

- `02_pybamm_model/closed_loop_model/configs/degradation_protocol.json`
- `02_pybamm_model/closed_loop_model/configs/model_parameter_overrides.json`
- `03_simulation/closed_loop/configs/calibration_map.json`
- `03_simulation/closed_loop/configs/loss_weights.json`

## Main Entrypoints

运行单次耦合退化仿真：

```bash
python 03_simulation/closed_loop/src/degradation/coupled_degradation_model.py
```

运行 RPT 校准：

```bash
python 03_simulation/closed_loop/src/calibration/rpt_calibration.py
```

运行完整闭环：

```bash
python 03_simulation/closed_loop/src/workflow/closed_loop_pipeline.py
```

## Main Outputs

- `03_simulation/closed_loop/results/degradation/`
- `03_simulation/closed_loop/results/calibration/`
- `03_simulation/closed_loop/results/workflow/`
- `03_simulation/closed_loop/results/baseline/`

## Notes

- 默认路径已经改到新结构，不再使用旧的根目录 `configs/`、`data/`、`models/`
- 基线 BOL 模型代码位于 `02_pybamm_model/closed_loop_model/src/baseline/`
