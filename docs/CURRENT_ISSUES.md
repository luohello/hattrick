# GEANT 基础实验状态

最后更新：2026-08-17

## 实验范围

本阶段只运行 Hattrick、SWAN 和 BEST_MC，不运行 DOTE、HARP 或消融实验。

- 数据集：GEANT，10772 个快照
- 划分：训练 `[0, 6464)`、验证 `[6464, 8618)`、测试 `[8618, 10772)`
- 预测器：ESM
- 候选路径：K=8（与论文一致）
- Hattrick：60 Epoch，batch size 64，RTX 4090
- 评价：测试集三类流量的 NormFulFill，并分别记录推理/求解时间

## 已完成的配置和工程修正

| 内容 | 涉及文件 |
| --- | --- |
| GEANT 从 K=4 改为论文使用的 K=8 | `run_geant_train_first.sh`、`run_geant_test_both.sh`、两个 PowerShell 基线脚本、结果汇总脚本 |
| Gurobi 使用 barrier、aggressive presolve、关闭 crossover | `frameworks/gurobi_refactored.py` |
| Gurobi 仅在数值状态 12 时启用稳定性重试，并累计全部求解时间 | `frameworks/gurobi_refactored.py` |
| K=8 与旧 K=4 的 SWAN 分流结果隔离 | `frameworks/gurobi_utils.py` |
| 训练集只加载一次，不再每个 epoch 重建 | `run_hattrick.py` |
| 异常检测改为按需开启，移除逐 batch CUDA cache 清理 | `run_hattrick.py`、`utils/training_utils.py` |
| 验证支持批处理，损失计算改为等价的张量运算 | `utils/args_parser.py`、`utils/training_utils.py` |
| 测试加载验证阶段选出的最佳模型，GPU 计时增加同步 | `run_hattrick.py` |
| 修正多 cluster 验证的 all-traffic 汇总变量 | `utils/training_utils.py` |

## 保留的一项差异

GEANT 三类流量继续使用当前划分比例，不重新生成数据。这与论文使用 PrivateWAN 比例拆分 GEANT 的方法不同，因此结果应表述为“论文其他配置对齐下的复现实验”，不能宣称为论文数值的完全复刻。

## 运行顺序

1. 生成 K=8 的 ground-truth oracle。
2. 上传 oracle 到远程服务器并训练 Hattrick。
3. 在同一测试区间运行 BEST_MC 和 SWAN。
4. 运行 Hattrick 的测试模拟并汇总三种方法。

实验数据、最优解、模型、日志和输出不提交到 Git，仅同步受版本控制的源码、脚本和文档。
