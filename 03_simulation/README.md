# 03 Simulation

这个板块放“给定模型与超参数后，真正跑仿真”的入口、数据、notebook 和结果。

统一骨架：

- 入口脚本：[main.py](/Z:/AlphaBattery/03_simulation/main.py)
- 输出目录：`03_simulation/outputs/`

## Contents

- `closed_loop/`
  - `src/degradation/`
  - `src/calibration/`
  - `src/workflow/`
  - `data/`
  - `configs/`
  - `notebooks/`
  - `results/`
- `reproduce_li2024/`
  - `scripts/run_from_json.py`
  - `notebooks/`
  - `outputs/`
  - `legacy/`
  - `slurm/`

## Main Entrypoints

```bash
python 03_simulation/main.py run-closed-loop
python 03_simulation/main.py run-reproduce
python 03_simulation/main.py export-tensorboard --target all
```

## Notes

- 闭环仿真的默认输出目录已经从旧的 `models/` 改为 `results/`
- Li2024 的默认输出目录保持为 `outputs/`
- 板块级 TensorBoard 日志目录是 `03_simulation/outputs/tensorboard/`
- 推荐命令：

```bash
tensorboard --logdir 03_simulation/outputs/tensorboard
```

远程服务器和 GPU 节点说明见：

- [cluster_usage.md](/Z:/AlphaBattery/01_requirements/cluster_usage.md)
