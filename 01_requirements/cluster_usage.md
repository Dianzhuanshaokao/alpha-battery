# Cluster Usage

## Purpose

本仓库的主要模拟与训练任务默认都在远程服务器上执行，而不是在本地桌面环境完成。

## Remote Server

- 服务器地址：`121.48.164.50`
- 主要用途：PyBaMM 仿真、强化学习训练、结果汇总、TensorBoard 查看
- 已验证可用用户：`zxsun`
- 本地可用密钥：`~/.ssh/id_rsa`

当前说明已基于一次实机验证更新。  
注意：如果直接从挂载盘路径使用 `Z:/.ssh/id_rsa`，`ssh` 可能会因为权限过宽而拒绝读取私钥。更稳妥的做法是先复制到本地目录并收紧 ACL 后再使用。

已验证连接方式：

```bash
ssh -i ~/.ssh/id_rsa zxsun@121.48.164.50
```

如果本地是 Windows 且私钥位于挂载盘，建议先复制到本地目录并限制为当前用户只读。

## Compute Types

- `c1`、`c2`、`c3`：GPUCompute 节点
- 其他节点：CPUCompute 节点

## GPU Usage

如果任务需要 GPU，提交作业或进入交互环境后，先加载 GPU 模块：

```bash
module load gpu
```

## Recommended Workflow

1. 通过 SSH 登录远程服务器
2. 根据任务类型选择 CPU 或 GPU 节点
3. 如果需要 GPU，先执行 `module load gpu`
4. 激活项目环境
5. 运行 `03_simulation` 或 `04_rl_optimization` 对应入口
6. 将结果写入板块级 `outputs/`
7. 使用 TensorBoard 查看训练或仿真日志

## Verified Conda Environment

当前远程机器上已验证可用的项目环境为：

- `pp`

推荐激活方式：

```bash
source ~/.bashrc
conda activate pp
```

已在该环境中补齐并验证导入的核心依赖包括：

- `pybamm`
- `pybtex`
- `tensorboard`
- `torch`
- `gymnasium`
- `stable_baselines3`
- `bayesian-optimization`
- `pyyaml`

## TensorBoard

推荐在远程服务器上启动 TensorBoard，然后通过 SSH 端口转发到本地浏览器查看。

仿真日志目录：

```bash
tensorboard --logdir 03_simulation/outputs/tensorboard --port 6006
```

RL 训练日志目录：

```bash
tensorboard --logdir 04_rl_optimization/outputs/training --port 6006
```

本地转发示例：

```bash
ssh -L 6006:127.0.0.1:6006 <user>@121.48.164.50
```

然后在本地浏览器打开：

```text
http://127.0.0.1:6006
```
