# Hattrick 基础实验与优化实验代码对照

最后更新：2026-08-22

## 1. 对照范围

本文档基于实际 Git 差异和已完成的 GEANT K=8 实验，比较：

- 基础版本：`main@f08435e`
- 优化版本：`optimized/phase1-phase2@e8b5a9b`
- 代码差异：13 个文件，新增 603 行、删除 214 行
- 数据集：GEANT，共 10772 个快照
- 训练区间：`[0, 6464)`
- 验证区间：`[6464, 8618)`
- 测试区间：`[8618, 10772)`，共 2154 个快照
- 候选路径数：K=8

两次正式实验都从头训练 60 Epoch，batch size 为 64，学习率为 0.0005，使用 ESM 预测 TM、三阶段 RAU、checkpoint 级别 2，并保持相同的数据划分、拓扑、候选路径和 Gurobi oracle。

需要特别说明：优化实验最终保留了论文代码原来的四个训练目标，没有启用后来实现的六级 Fulfill 目标。因此，本文把“已写入代码”和“正式实验实际启用”分开记录。

## 2. 正式实验配置差异

| 配置项 | 基础实验 | 优化实验 | 是否影响正式结果 |
|---|---|---|---|
| 训练目标 | 原始四目标 | 原始四目标 | 保持一致 |
| 梯度投影实现 | 原始手工分支实现 | `autograd.grad + SVD` | 是 |
| TM 不确定性校准 | 无 | 因果低估残差 EMA | 是 |
| CVaR 尾部损失 | 无 | `alpha=0.1`，权重 0.1 | 是 |
| 链路编码 | 源宿节点表示求和 + 原始容量 | 源、宿、方向差 + 归一化容量 | 是 |
| RAU 次数 | 每阶段固定 3 次 | 每阶段最多 3 次，可提前停止 | 是 |
| RAU 停止阈值 | 无 | `max(abs(delta_logit)) < 0.001` | 是 |
| 确定性算法 | 硬编码开启 | 参数化，正式实验开启 | 运行语义一致 |
| 条件式六级 Fulfill | 无 | 代码保留，但 `conditional_fulfill=0` | 否 |
| 训练/测试资产 | 与基础结果同目录 | 独立目录，只读复用输入和 oracle | 影响实验管理，不改变算法 |

正式优化入口见 [`run_geant_optimized.sh`](../run_geant_optimized.sh)，其中关键参数为：

```text
directional_edge_encoding = 1
adaptive_rau_tol          = 0.001
adaptive_rau_min_steps    = 1
uncertainty_scale         = 1.0
uncertainty_ema           = 0.9
cvar_alpha                = 0.1
cvar_weight               = 0.1
conditional_fulfill       = 0
```

## 3. 保留论文原始四目标

基础版和最终优化实验都使用以下四个优先目标：

1. `loss1`：高优先级 MLU
2. `loss2`：高+中优先级累计 MLU
3. `loss3`：三类总接纳流量 / MaxFlow
4. `loss4`：三类累计 MLU

优化分支曾实现条件式六级目标：

```text
Fh -> MLUh -> Fhm -> MLUhm -> Fhml -> MLUhml
```

但该实现会改变论文原始目标结构，而且代码中的第五级还组合了 `fulfill_123 + loss3`。为避免在第一轮优化实验中混入额外目标语义，提交 `3b6c37d` 将正式入口改为 `--conditional_fulfill 0`。六级目标相关函数仍保留在 [`utils/training_utils.py`](../utils/training_utils.py)，只用于后续单独消融，不属于本次优化结果。

## 4. 稳定的优先级梯度投影

### 4.1 基础实现

基础实现依次对四个 loss 调用 `backward`，读取每个参数的 `.grad`，再通过多个针对 1、2、3、4 个有效梯度的条件分支完成投影。其主要问题是：

- 依赖宽泛的 `try/except` 处理缺失梯度；
- 对零梯度和线性相关梯度存在较多特殊分支；
- 遇到 NaN 时直接 `exit(1)`；
- 每次参数更新后调用 `torch.cuda.empty_cache()`；
- 代码复杂，难以验证所有退化情况。

### 4.2 优化实现

[`utils/robust_proj_utils.py`](../utils/robust_proj_utils.py) 被重写为统一流程：

1. 使用 `torch.autograd.grad` 分别计算每个目标对所有可训练参数的梯度；
2. 使用 `allow_unused=True`，对未参与某目标的参数填充零梯度；
3. 将每个目标的梯度展平，范数超过 25 时进行裁剪；
4. 把更高优先级的非零梯度堆叠成矩阵；
5. 使用 reduced SVD 和数值秩阈值构造受保护子空间；
6. 将低优先级梯度投影到该子空间的正交补；
7. 检查非有限值和最终梯度长度，再显式写回参数并执行优化器更新。

形式上，第 `k` 个低优先级梯度 `g_k` 被处理为：

```text
g_k_projected = g_k - U(U^T g_k)
```

其中 `U` 是前面高优先级有效梯度张成子空间的正交基。这样保留了原始四目标的优先级思想，但减少了特殊分支，并能更稳定地处理零梯度和线性相关梯度。

此外，优化版删除了逐批 `torch.cuda.empty_cache()`。这避免了频繁触发 CUDA 分配器同步，但本次实验没有单独记录峰值显存，因此暂时不能量化显存变化。

## 5. 因果不确定性感知 TM 校准

基础模型直接使用 ESM 给出的点预测 TM。优化版在 [`utils/build_dataset_within_cluster.py`](../utils/build_dataset_within_cluster.py) 中加入低估残差的因果 EMA。

对每个流量类别和 OD 元素，时刻 `t` 的处理近似为：

```text
adjusted_prediction[t] = raw_prediction[t] + scale * margin[t]
under_prediction[t]    = max(actual[t] - raw_prediction[t], 0)
margin[t+1]            = ema * margin[t] + (1-ema) * under_prediction[t]
```

正式配置使用：

```text
scale = 1.0
ema   = 0.9
```

关键因果边界是：当前快照先使用此前残差形成的 `margin[t]` 调整预测，随后才把当前真实流量的低估残差更新到 `margin[t+1]`。因此当前预测不读取当前或未来真实 TM。

该方法针对的是 ESM 低估流量导致的容量准备不足，代价是需要假设上一时刻真实流量已经可观测。训练、验证和测试 Dataset 分别初始化自己的 EMA 状态，因此每个数据区间的第一个快照都从零 margin 开始。

## 6. CVaR 风格的尾部 MLU 优化

优化版在 [`utils/training_utils.py`](../utils/training_utils.py) 中增加 `tail_mlu_loss`。对 batch 内每个样本先计算相对 Gurobi 最优值的归一化 MLU，然后选取最差的 `ceil(batch_size * alpha)` 个样本求均值。

正式配置为 batch size 64、`alpha=0.1`，因此每个 batch 约选择最差 7 个样本。尾部损失以 0.1 的权重加入：

- 高优先级 MLU `loss1`
- 高+中累计 MLU `loss2`
- 三类累计 MLU `loss4`

原始 MaxFlow 目标 `loss3` 不增加 CVaR 项。这样仍保持四目标顺序，但每个 MLU 目标会额外关注 batch 尾部拥塞，而不是只优化普通均值。

## 7. 方向感知链路编码

基础版的链路表示为：

```text
[h_src + h_dst, capacity]
```

源、宿节点 embedding 相加后，反向链路很难从表示中区分。优化版在 [`frameworks/hattrick_system.py`](../frameworks/hattrick_system.py) 中改为：

```text
[h_src, h_dst, h_src - h_dst, capacity / mean(abs(capacity))]
```

具体改进包括：

- 显式保留源节点和目的节点角色；
- 使用 `h_src-h_dst` 表示方向；
- 用当前图的平均绝对容量归一化链路容量；
- 根据新的输入维度自动寻找可以整除 embedding 维度的 Transformer head 数；
- 若用户指定的 head 数不能整除输入维度，直接抛出明确错误；
- 创建 batch 索引时显式使用 node embedding 所在设备，避免 CPU/GPU 索引设备不一致。

该改动增加了 Transformer 输入维度，因此它不只是无参数的特征重排；优化模型必须重新训练，不能直接复用基础模型权重。本次实验也尚未完成“等参数量方向编码”消融。

## 8. 自适应 RAU 早停

基础实验在三个阶段分别固定执行 3 次 RAU。优化版仍保留 3 次作为上限，但每次计算 `delta_gammas` 后检查：

```text
max(abs(delta_gammas)) < adaptive_rau_tol
```

正式阈值为 0.001，最少迭代次数为 1。满足条件时当前阶段提前退出；否则最多执行 3 次。

模型会把最近一次 forward 的实际执行次数保存在：

```text
model.last_rau_steps = [stage1_steps, stage2_steps, stage3_steps]
```

但测试循环没有把该值逐快照写入文件，汇总脚本也没有统计平均 RAU 次数。因此目前只能确认早停逻辑存在，无法从已完成实验中确认它实际触发了多少次。

此外，收敛判断使用 `.item()` 读取 GPU 张量，会在每个 RAU 迭代处引入一次 CPU/GPU 同步。这可能抵消部分早停收益，也是优化版推理时间没有下降的可能原因之一。

## 9. 训练与运行工程改进

### 9.1 可配置运行开关

[`utils/args_parser.py`](../utils/args_parser.py) 新增了以下参数：

- `deterministic`
- `directional_edge_encoding`
- `adaptive_rau_tol`
- `adaptive_rau_min_steps`
- `uncertainty_scale`
- `uncertainty_ema`
- `conditional_fulfill`
- `fulfill_slo`
- `cvar_alpha`
- `cvar_weight`

基础版把 PyTorch deterministic algorithms 硬编码开启；优化版改为参数控制，但正式实验仍设置为 1，因此两次正式实验都使用确定性内核。

### 9.2 统一入口和失败检查

新增 [`run_geant_optimized.sh`](../run_geant_optimized.sh)，提供：

```bash
./run_geant_optimized.sh check
./run_geant_optimized.sh smoke
./run_geant_optimized.sh train
./run_geant_optimized.sh test
./run_geant_optimized.sh all
```

入口脚本会执行：

- Python 语法编译检查；
- 优化组件单元测试；
- 必需 oracle/文件资产检查；
- 正式训练；
- MLU 与 MaxFlow/Fulfill 两次推理；
- 输出行数校验；
- 结果汇总和图表生成；
- 失败行号与独立日志记录。

[`run_geant_optimized_test.sh`](../run_geant_optimized_test.sh) 会验证每次推理产生 2154 行 runtime 和 6462 行三类结果，降低半截结果被误当作完整实验的风险。

### 9.3 实验资产隔离

实验运行阶段曾使用 `/mnt/data0/Hattrick_optimized` 隔离优化 worktree。优化代码合并到 `main` 后，远程统一使用 `/mnt/data0/Hattrick`，并采用命名子目录保留两套实验资产：

- 继续复用同一份 GEANT TM、ESM TM、filenames、Gurobi oracle、BEST-MC 和 SWAN 结果；
- 基础 Hattrick 模型和原始结果归档到 `baseline_geant_k8` / `baseline_hattrick` 子目录；
- 优化 Hattrick 模型和原始结果归档到 `optimized_geant_k8` / `optimized_hattrick` 子目录；
- 活动代码、活动模型和默认 Hattrick 结果使用优化版本；
- 基础汇总与优化汇总分别保存在 `output/geant_k8` 和 `output/geant_k8_optimized`；
- 数据、模型和大日志不提交 Git，只提交源码、脚本、文档和结果摘要。

由于 BEST-MC、SWAN 和 Gurobi oracle 是只读复用的，基础版与优化版之间真正变化的是 Hattrick；其他方法作为相同参照，不应被描述为重新优化后的结果。

### 9.4 自动排队

新增 [`run_after_baseline.sh`](../run_after_baseline.sh)。它等待基础日志出现 `GEANT K=8 pipeline completed`，检查优化实现放行标记后自动执行完整优化流水线。这样避免基础与优化训练同时争用实验资源。

## 10. 验证覆盖

[`tests/test_optimized_components.py`](../tests/test_optimized_components.py) 当前包含 3 项单元测试：

1. 低优先级投影分量与高优先级梯度正交；
2. 因果不确定性校准不使用当前快照真实值；
3. Fulfill SLO hinge 在违反阈值时产生正确方向的梯度。

远程还完成了语法检查、两样本一轮冒烟训练、验证前向传播和正式 60 Epoch 流水线。

尚未单独覆盖：

- 零梯度和多组线性相关梯度；
- 方向编码对正反向边的可区分性；
- 三阶段 RAU 提前停止边界；
- CVaR top-k 的边界值；
- 四目标新旧投影在固定输入上的数值对照；
- 不同随机种子的重复实验。

## 11. 当前实测结果概览

以下结果来自同一远程服务器、同一测试区间的单次训练。Fulfill/MaxFlow 和 MLU 均使用 Gurobi oracle 归一化，1.0 表示接近 oracle；略高于 1 的数值可能来自数值容差。

| 指标 | 基础版 | 优化版 | 变化 |
|---|---:|---:|---:|
| 中优先级 Fulfill Mean | 0.999473 | 0.999826 | +0.0352 个百分点 |
| 中优先级 Fulfill P1 | 0.991082 | 0.993943 | +0.2862 个百分点 |
| 中优先级 Fulfill P10 | 0.998133 | 0.999425 | +0.1292 个百分点 |
| 低优先级 Fulfill P1 | 0.986009 | 0.983092 | -0.2917 个百分点 |
| 三类总 MaxFlow Mean 达成率 | 1.000344 | 1.000526 | +0.0182 个百分点 |
| 高优先级归一化 MLU Mean | 1.002658 | 1.002240 | 更接近最优 |
| 高+中归一化 MLU Mean | 1.001868 | 1.000942 | 更接近最优 |
| 三类累计归一化 MLU Mean | 1.001996 | 1.000906 | 更接近最优 |
| MaxFlow 模式平均推理时间 | 109.041 ms | 109.789 ms | +0.686%，变慢 |
| MLU 模式平均推理时间 | 108.459 ms | 109.305 ms | +0.780%，变慢 |

现阶段可以得出的结论是：

- 中优先级完成率及大多数平均/P95 MLU 指标改善；
- 三类总 MaxFlow 小幅提升；
- 低优先级最差 1% 完成率退化，说明极端低优先级快照仍需处理；
- 推理稳定时延增加约 0.7% 至 0.9%，尚未形成加速；
- 自适应 RAU 的实际触发次数未记录，因此不能把时延变化直接归因于 RAU；
- 当前只有一个 seed，不能宣称改进具有统计稳定性。

远程结果目录：

```text
基础版：/mnt/data0/Hattrick/output/geant_k8
优化版：/mnt/data0/Hattrick/output/geant_k8_optimized
```

## 12. 总结

本次优化没有改变 GEANT 数据、K=8 候选路径、训练/验证/测试区间、网络主体配置和论文原始四目标顺序。核心改动集中在：

1. 用统一的 `autograd.grad + SVD` 重写优先级梯度投影；
2. 使用仅依赖历史信息的低估残差 EMA 校准 ESM TM；
3. 用 CVaR 风格损失关注尾部 MLU；
4. 用方向感知、容量归一化的链路表示替代对称求和表示；
5. 为三阶段 RAU 增加按 logit 更新幅度早停；
6. 增加独立运行、资产校验、冒烟测试、结果隔离和自动排队。

从当前单次实验看，优化主要换来了更好的中优先级尾部完成率和更接近 Gurobi 最优值的累计 MLU，但尚未带来推理加速，并伴随低优先级 P1 退化。下一步应优先补充 RAU/显存采集、低优先级异常快照分析和多 seed 消融，而不是直接将当前结果描述为全面优于基础版本。
