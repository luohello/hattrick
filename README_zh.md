# Hattrick：使用神经模型解决多类流量工程

[Hattrick](https://doi.org/10.1145/3718958.3750470) 是一种用于 WAN 多类流量工程（Multi-Class Traffic Engineering）的可迁移神经网络，目标是在考虑预测误差的同时联合优化多类流量。该工作发表于 ACM SIGCOMM 2025。

英文原始说明见 [README.md](README.md)。本文件用于记录项目的中文运行说明；论文复现中发现的问题与后续改进计划分别见：

- [当前问题记录](docs/CURRENT_ISSUES.md)
- [改进优化计划](docs/IMPROVEMENT_PLAN.md)

## 环境

作者验证环境：

- Rocky Linux 9.5
- Python 3.12.5
- `torch==2.5.1+cu124`
- `torch-scatter==2.1.2+pt25cu124`

当前项目远程环境位于数据盘 `/mnt/data0`：

```bash
cd /mnt/data0/Hattrick
source /mnt/data0/helo/bin/activate
```

其余依赖参见：

- `requirements.txt`：作者原始依赖；
- `requirements-helo.txt`：当前远程训练环境补充依赖；
- `requirements-gurobi.txt`：本地 Gurobi 求解环境依赖。

## 完整运行链路

```text
准备 topology / pairs / traffic matrices / manifest
  -> 生成预测 TM（ESM 或其他预测器）
  -> 用 Gurobi 生成最优 MLU / MaxFlow
  -> 运行 BEST_MC 和 SWAN 基线
  -> 训练 Hattrick
  -> 分别执行 MLU 与 Fulfill Ratio 测试
  -> 汇总 CSV、表格和图像
```

## 数据格式

Manifest 中每行包含：

```text
topology_file.json,pairs_file.pkl,traffic_matrix.pkl
```

流量矩阵应为 `(num_pairs, 1)` 数组，并与 pairs 中的 OD 顺序严格对应：

```text
traffic_matrices/<topology>_1/       高优先级真实 TM
traffic_matrices/<topology>_2/       中优先级真实 TM
traffic_matrices/<topology>_3/       低优先级真实 TM
traffic_matrices/<topology>_1_esm/   高优先级 ESM 预测 TM
traffic_matrices/<topology>_2_esm/   中优先级 ESM 预测 TM
traffic_matrices/<topology>_3_esm/   低优先级 ESM 预测 TM
```

数据集、预测矩阵、模型权重、日志和批量结果不提交到普通 Git 仓库。

## GEANT 当前复现

当前已完成：

- GEANT 单类流量拆分及 ESM 预测；
- K=4 下的 Gurobi oracle；
- BEST_MC 和 SWAN 基线；
- Hattrick 60 Epoch 训练；
- 2154 个测试快照的 MLU 和 Fulfill Ratio 推理；
- CSV、CDF、箱线图和报告生成。

当前训练入口：

```bash
bash run_geant_train_first.sh
```

当前两次推理入口：

```bash
bash run_geant_test_both.sh
```

结果汇总：

```bash
python summarize_geant_results.py --output-dir /mnt/data0/Hattrick/output
```

主要结果位于：

```text
/mnt/data0/Hattrick/results/geant/4sp/0/
/mnt/data0/Hattrick/output/
```

## Gurobi

Gurobi 不在每个神经网络训练 batch 内运行。它用于：

1. 离线计算 ground-truth TM 下的最优 MLU 和 MaxFlow；
2. 运行 BEST_MC（代码中的 `flexile`）和 SWAN 基线；
3. 为测试指标提供归一化参考。

本地 Windows 批量脚本：

```powershell
.\run_geant_gurobi_full.ps1
.\run_geant_baselines.ps1
```

## 公开数据集

- GEANT：作者提供预处理的单类 TM，需要进一步拆成多类别；
- USCarrier：作者提供合成的三类别流量矩阵；
- KDL：作者提供合成的三类别流量矩阵；
- Abilene：仓库只提供数据准备示例，论文没有使用该数据集评估 Hattrick。

作者数据链接及原始命令以 [README.md](README.md) 为准。

## 关键命令行参数

| 参数 | 含义 |
| --- | --- |
| `--topo` | 拓扑/数据集名称 |
| `--mode` | `train` 或 `test` |
| `--num_paths_per_pair` | 每个 OD 的候选路径数 K |
| `--rau1/2/3` | 三阶段 RAU 迭代次数 |
| `--pred` | 是否使用预测 TM |
| `--pred_type` | 预测器名称，如 `esm` |
| `--dynamic` | 拓扑是否随快照变化 |
| `--violation` | 是否启用 Violation MLP |
| `--checkpoint` | 梯度检查点等级 0/1/2 |
| `--gur_mode` | Gurobi 基线模式：`flexile` 或 `swan` |
| `--sim_mf_mlu` | 测试 Fulfill Ratio/模拟器路径或 MLU 路径 |

注意：当前代码中 `initial_training=1` 会创建新模型，`initial_training=0` 才尝试恢复已有模型；这与作者 README 的文字说明存在歧义，后续计划统一语义。

## 当前限制

- 当前 GEANT 使用 K=4，而论文使用 K=8；
- 当前 GEANT 类别拆分比例需要进一步与论文核对；
- Hattrick 与 Gurobi 基线运行时间来自不同硬件，暂不能作为严格受控加速比；
- 当前测试默认加载最后一轮模型，后续将增加明确的 best/last checkpoint 选择；
- 训练与验证存在可优化的 I/O、CUDA cache 和 batch 处理开销。

详细内容见 `docs/CURRENT_ISSUES.md` 和 `docs/IMPROVEMENT_PLAN.md`。
