# 02 PyBaMM Model

这个板块放 PyBaMM 模型相关资产，不直接承载主要结果输出。

统一骨架：

- 入口脚本：[main.py](/Z:/AlphaBattery/02_pybamm_model/main.py)
- 输出目录：`02_pybamm_model/outputs/`

## Contents

- `closed_loop_model/`
  - `src/baseline/`: BOL/热模型基线代码
  - `configs/`: 闭环模型协议与参数覆盖
  - `scripts/build_local_idaklu.py`: 本地 solver 构建辅助脚本
- `reproduce_li2024_model/`
  - `sim/`: Li2024 复现模型核心
  - `configs/cases/`: Li2024 案例配置

## Typical Use

闭环基线模型会把结果写到本板块的 `outputs/baseline/`：

```bash
python 02_pybamm_model/main.py run-baseline
```

Li2024 模型核心由第三板块的运行脚本调用，不建议直接从 `sim/` 手动拼装入口。

远程执行和节点选择规则见：

- [cluster_usage.md](/Z:/AlphaBattery/01_requirements/cluster_usage.md)
