# Hattrick 改进优化计划

最后更新：2026-08-17

本文档记录拟进行的代码和算法改进、具体涉及文件、验证方法与推荐实施顺序。问题背景与当前证据见 `docs/CURRENT_ISSUES.md`。

## 1. 总体原则

1. 先建立可信基线，再改变算法。
2. 每次实验只改变一个主要因素，并保留配置与 Git commit。
3. 主要评价尾部性能（P1、P10），同时报告 Median、Mean、运行时间和峰值显存。
4. 高优先级指标优先；中、低优先级的改进不能以明显损害高优先级为代价。
5. 数据、权重和日志不进入普通 Git，结果通过结构化摘要和外部实验目录保存。

## 2. Phase 0：修正复现与评测口径

### OPT-001：可配置、可追踪的 GEANT 类别拆分

- 对应问题：ISSUE-001
- 修改文件：
  - `prepare_geant/prepare_geant_hattrick.py`
  - 建议新增 `configs/data/geant_split_*.json`
- 计划改动：
  - 增加 `--split-strategy`、`--seed` 和比例配置文件；
  - 支持固定类别比例、PrivateWAN 风格比例以及当前 synthetic 比例；
  - 输出拆分元数据和每类比例统计；
  - 保证断点续跑不改变随机序列。
- 验证：三类矩阵逐元素之和等于原始 TM；相同 seed 结果完全一致；不同策略的比例分布符合配置。

### OPT-002：论文配置基线 K=8

- 对应问题：ISSUE-002
- 修改文件：
  - `run_geant_train_first.sh` 或新增 `scripts/run_geant_k8_train.sh`
  - Gurobi/基线启动脚本
  - 结果汇总配置
- 计划改动：重新生成 KSP、Gurobi oracle、BEST_MC、SWAN 和 Hattrick 权重，不覆盖现有 K=4 结果。
- 验证：输出目录和模型名明确包含 K、seed、split ID；同一实验的所有方法使用相同路径集合。

### OPT-003：统一 best/last/checkpoint

- 对应问题：ISSUE-003、ISSUE-004
- 修改文件：
  - `run_hattrick.py`
  - 建议新增 `utils/checkpoint_utils.py`
- 计划改动：
  - 保存 `state_dict`、优化器状态、epoch、配置和最佳验证指标；
  - 用词典序比较器选择 best；
  - 测试增加 `--checkpoint best|last|<path>`；
  - 不再保存完整 Python 模型对象。
- 验证：中断恢复后下一 epoch 与连续训练一致；测试日志明确写出 checkpoint 路径和 epoch。

### OPT-004：可信运行时间基准

- 对应问题：ISSUE-005
- 修改文件：
  - `run_hattrick.py`
  - `frameworks/gurobi_refactored.py`
  - `summarize_geant_results.py`
- 计划改动：CUDA Event/synchronize 计时；预热若干样本；分别输出冷启动、稳态中位数、P95 和端到端时间；记录硬件信息。
- 验证：重复测量方差合理；相同计时区间不包含或同时包含数据加载。

## 3. Phase 1：不改变算法语义的工程优化

### OPT-101：训练调试开关和显存管理

- 对应问题：ISSUE-007
- 修改文件：`run_hattrick.py`、`utils/training_utils.py`、`utils/args_parser.py`
- 计划改动：将 anomaly detection、deterministic algorithms 和 CUDA cache 清理改为可配置；默认训练不逐 batch 清缓存；比较 checkpoint 0/1/2。
- 验证：相同 seed 下指标处于数值容差内；记录每 batch 时间和峰值显存。

### OPT-102：批量验证和向量化损失

- 对应问题：ISSUE-007、ISSUE-009
- 修改文件：`utils/training_utils.py`、`run_hattrick.py`
- 计划改动：
  - 验证 batch size 改为可配置；
  - 移除 loss 内逐样本 `.item()`；
  - 修复多 cluster all-traffic 聚合；
  - 使用张量一次计算 batch 指标。
- 验证：新旧实现对同一固定输出的指标一致；验证耗时显著下降。

### OPT-103：数据集复用和惰性加载

- 对应问题：ISSUE-007
- 修改文件：
  - `utils/build_dataset_within_cluster.py`
  - `utils/snapshot_utils.py`
  - `run_hattrick.py`
- 计划改动：Dataset 只构造一次；静态拓扑和 pairs 只读一次；TM 按需加载或使用内存映射；不在文件加载阶段将需求复制 K 份，改为计算时广播。
- 验证：随机抽样核对新旧 batch 内容；记录初始化时间、CPU 内存和 epoch 时间。

### OPT-104：稳定且高效的梯度投影

- 对应问题：ISSUE-008
- 修改文件：`utils/robust_proj_utils.py`、新建 `tests/test_gradient_projection.py`
- 计划改动：使用 `torch.autograd.grad` 取得多目标梯度；使用 QR/SVD 构造受保护子空间；移除 broad exception 和 `exit(1)`；记录梯度范数、夹角和被投影比例。
- 验证：投影后正交误差低于阈值；零梯度和线性相关梯度不会崩溃；与当前实现做收敛消融。

## 4. Phase 2：推荐的核心算法改进

### OPT-201：Uncertainty-Aware Hattrick

- 目标：让模型不仅看到点预测，还知道预测的不确定程度，重点改善 Medium/Low 的 P1 和 P10。
- 修改文件：
  - `traffic_matrices/esm_predictor.py`
  - `utils/snapshot_utils.py`
  - `frameworks/hattrick_system.py`
  - `utils/training_utils.py`
  - 建议新增 `utils/uncertainty_utils.py`
- 输入扩展：为每个 OD/类别加入历史残差标准差、低估偏差、最近误差和变化率等仅依赖过去信息的特征。
- 训练扩展：从训练期历史残差采样多个可能 TM 场景，优化尾部风险或 CVaR；推理仍不读取未来真实 TM。
- 消融：
  - A：原始 Hattrick；
  - B：仅增加不确定性输入；
  - C：仅增加尾部/CVaR 损失；
  - D：两者同时启用。
- 成功标准：高优先级 P1 不显著退化，Medium/Low P1 或 P10 稳定提升，并在至少三个 seed 上成立。

### OPT-202：条件式恢复高/高+中 Fulfill 目标

- 目标：去除“高优先级总能完全满足”的强假设，提高突发流量和链路故障下的稳健性。
- 修改文件：`utils/training_utils.py`、`utils/robust_proj_utils.py`
- 计划改动：当 Fulfill Ratio 低于 SLO 阈值时启用 hinge loss，并按 `Fh -> MLUh -> Fhm -> MLUhm -> Fhml -> MLUhml` 顺序投影。
- 验证：正常场景不引入明显额外梯度；压力场景和故障场景中高优先级尾部更稳定。

### OPT-203：方向感知的链路编码

- 目标：避免当前源、宿节点 embedding 相加导致方向信息弱化。
- 修改文件：`frameworks/hattrick_system.py`
- 计划改动：将链路表示改为 `[h_src, h_dst, h_src-h_dst, normalized_capacity]`，并对容量和需求使用无量纲归一化。
- 验证：与当前 sum encoder 做等参数量消融；检查反向链路容量不同时的表示可区分性。

### OPT-204：自适应 RAU 早停

- 目标：对简单快照减少迭代，对困难快照保留完整调整次数，形成可控的性能-时延折中。
- 修改文件：`frameworks/hattrick_system.py`、`utils/args_parser.py`、`summarize_geant_results.py`
- 早停信号：MLU 改善量、logit 更新范数或分流比例变化量低于阈值。
- 验证：报告平均 RAU 次数、P95 推理时间和 Fulfill Ratio 尾部变化；优先在 USCarrier 上验证扩展性。

## 5. Phase 3：后续研究方向

### OPT-301：时延感知多类别 TE

- 修改文件：拓扑格式、路径预处理、`hattrick_system.py`、模拟器和损失函数。
- 思路：为链路/隧道加入时延，高优先级仅允许低时延隧道或增加时延 SLO。
- 前置条件：获得可靠链路时延，或明确说明合成时延生成方法。

### OPT-302：候选路径多样性与学习式剪枝

- 修改文件：`utils/cluster_utils.py`、`frameworks/hattrick_system.py`
- 思路：从单纯 K 最短路径扩展到容量感知、边不相交路径；在大拓扑中学习保留少量有效路径。
- 目标：在不降低尾部性能的情况下减少 KDL/USCarrier 的显存和推理时间。

## 6. 自动化测试计划

建议新增：

```text
tests/
├── test_data_alignment.py
├── test_split_ratios.py
├── test_simulator_capacity.py
├── test_gradient_projection.py
├── test_checkpoint_selection.py
└── test_metrics_schema.py
```

最低不变量：

1. 每个 OD、每个类别的 K 条路径分流比例之和为 1；
2. 模拟后的每类流量非负，累计链路负载不超过容量容差；
3. 高优先级先于中、低优先级消耗容量；
4. 投影后的低优先级梯度与受保护梯度空间近似正交；
5. manifest、pairs、真实 TM 和预测 TM 数量及顺序一致；
6. P1/P10/Median/P95 的方向和定义在所有拓扑上统一。

## 7. 推荐执行顺序

1. OPT-001～004：修正数据、K、checkpoint 和计时。
2. OPT-101～104：缩短训练周期并提高代码可靠性。
3. 在修正后的 GEANT K=8 上建立三个 seed 的稳定基线。
4. OPT-201：实现不确定性感知和尾部风险优化。
5. OPT-202～204：依次消融条件式目标、方向编码和自适应 RAU。
6. 在 USCarrier 上验证扩展性，在 KDL 或故障场景上做最终外推。

每个完成的优化应在本文档中补充：实现 commit、配置文件、结果目录、关键指标、是否保留以及回滚方法。
