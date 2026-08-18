# Learn2Branch 论文复现说明

本目录以论文作者发布的官方仓库为基线，目标是复现 *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*（NeurIPS 2019，arXiv:1906.01629）。上游代码固定在提交 `57e82603fba39c34ff81baf9bd0b5768f5a1a860`，本地开发分支为 `agent/reproduce-paper`。

## 已锁定的软件栈

- Conda 环境名：`learn2branch`
- Python 3.6.13、NumPy 1.16.6、SciPy 1.2.1
- TensorFlow GPU 1.12.0（当前机器验证时使用 CPU）
- scikit-learn 0.20.2
- SoPlex 4.0.1
- SCIP 6.0.1，并应用仓库中的 `vanillafullstrong.patch`
- 修改版 PySCIPOpt：`9b88eddcfe99208e737236d39ccd6c9dd1a9bec4`
- pyltr：`78fa0ebfef67d6594b8415aa5c6136e30a5e3395`
- PySVMRank：`b7dd8598d459ddb8926467385f5658173af46752`

SCIP、SoPlex 和 SVMrank 的归档都使用 `reproduction/checksums.sha256` 校验。构建源码和二进制位于被 Git 忽略的 `.repro/`，实验数据、模型和结果沿用官方目录 `data/`、`trained_models/`、`results/`。

## 创建和验证环境

环境创建是唯一必须从环境外发起的步骤：

```bash
cd /netfs/tsp/student/2023/zguo1/coding/evoscip_baseline/Learn2Branch
bash reproduction/create_environment.sh
conda activate learn2branch
bash reproduction/build_dependencies.sh
source reproduction/activate.sh
python reproduction/verify_environment.py
```

之后的下载、编译、数据生成、训练、测试和 GitHub 操作均在 `learn2branch` 环境中执行。`reproduction/activate.sh` 会拒绝其他环境，并设置本地 SCIP 动态库路径。

当前主机的 NVIDIA L40S/驱动远新于 TensorFlow 1.12 所要求的 CUDA 9.x 时代运行时，因此默认清空 `CUDA_VISIBLE_DEVICES`，已验证 CPU eager execution 和 SCIP 求解均可运行。这保持论文的软件版本，但训练硬件与论文环境不完全相同。

## 官方实验规模

官方 `01_generate_instances.py` 对每类问题生成 10,000 个训练实例、2,000 个验证实例、2,000 个测试实例，以及 small/medium/big 各 100 个迁移实例。`02_generate_dataset.py` 使用 SCIP `pscost` 探索、以 0.05 概率查询 patched `vanillafullstrong` 专家，生成 100,000/20,000/20,000 个训练/验证/测试样本。

GCNN 使用 5 个随机种子、batch size 32、每 epoch 312 个 batch、学习率 0.001、最多 1,000 epochs；最终求解评估使用每个尺度 20 个实例、5 个种子及 3,600 秒时限。该规模需要大量 CPU 时间和磁盘空间，因此仓库还会提供使用相同生成器、特征、专家和模型代码的端到端冒烟复现入口，然后再启动全量实验。

注意：上游 README 中的 `02_generate_samples.py` 是文件名笔误，实际脚本为 `02_generate_dataset.py`。

## 端到端冒烟复现

冒烟流程直接复用官方四个阶段的函数和类，不维护算法分叉：使用原始 set-cover 尺寸、`pscost` 探索、0.05 专家查询概率、baseline GCNN 和官方在线分支回调，仅将实例数、样本数和训练 epoch 缩小。

```bash
conda activate learn2branch
source reproduction/activate.sh
bash reproduction/run_smoke.sh
```

大体积中间文件写入 `reproduction_artifacts/smoke_setcover/`；可提交的指标摘要写入 `reproduction/results/smoke_setcover.json`。冒烟指标只证明端到端链路可运行，不能与论文全量结果比较。

## 全量复现入口

以下命令严格调用官方规模和超参数，并在 CPU 上依次运行四类问题的实例生成、专家采样、五随机种子训练、离线测试和在线求解：

```bash
bash reproduction/run_full.sh --problem all --phase all --jobs 4
```

建议按阶段运行以便检查磁盘和日志，例如：

```bash
bash reproduction/run_full.sh --problem setcover --phase instances --jobs 4
bash reproduction/run_full.sh --problem setcover --phase samples --jobs 32
bash reproduction/run_full.sh --problem setcover --phase train
bash reproduction/run_full.sh --problem setcover --phase test
bash reproduction/run_full.sh --problem setcover --phase evaluate
```

官方脚本会在目标目录已存在时主动报错，避免静默混合两次运行。因此每个生成阶段应只启动一次；继续已完成的流水线时从下一个 `--phase` 开始。

## 当前全量进度

四类问题的官方全量实例已经使用 seed 0 生成完毕：每类 14,300 个，共 57,200 个 LP 文件。可使用以下命令重新验证目录计数、连续文件名，并计算路径与内容的聚合 SHA-256：

```bash
python reproduction/audit_full_instances.py
```

审计结果保存在 `reproduction/results/full_instances_manifest.json`。实例本身保存在被 Git 忽略的 `data/instances/`，不会把数十 GB 数据错误提交到代码仓库。

全量专家采样使用集中式顺序队列，以避免最慢问题在 20-worker 并发方案中形成数十小时长尾：

```bash
LEARN2BRANCH_SAMPLE_JOBS=80 bash reproduction/run_full_sampling_queue.sh
```

默认顺序为 indset、facilities、setcover、cauctions。每类仍直接调用官方 `02_generate_dataset.py`，只改变并行 worker 数；完成标记和日志保存在 `reproduction_artifacts/full/`。若运行中断，脚本会拒绝在无完成标记的部分样本目录上继续，避免静默混合两次采样。

每类样本完成后，可增量审计精确计数、连续文件名、字节数及路径与内容的聚合 SHA-256：

```bash
python reproduction/audit_full_samples.py --problem indset
```

重复 `--problem` 可一次审计多类；不传时审计全部四类。结果会增量合并至 `reproduction/results/full_samples_manifest.json`，样本本身仍保存在被 Git 忽略的 `data/samples/`。

全量样本完成后，可在 96 核节点上并行启动官方的 5-seed 训练组合。以下两个队列使用不重叠的 CPU 亲和区间，可同时运行：

```bash
LEARN2BRANCH_GCNN_JOBS=8 bash reproduction/run_full_training_queue.sh gcnn
LEARN2BRANCH_COMPETITOR_JOBS=12 bash reproduction/run_full_training_queue.sh competitors
```

GCNN 队列包含四类问题的 baseline，以及 setcover 的 mean_convolution 和 no_prenorm，共 30 个训练；传统模型队列包含 ExtraTrees、SVMRank 和 LambdaMART，共 60 个训练。所有超参数与官方脚本保持一致。SVMRank 的单进程峰值内存约为 30 GiB，因此传统模型默认限制为 12 路并发。外部日志与可恢复完成标记保存在 `reproduction_artifacts/full_training/`；若发现无完成标记的部分模型目录，队列会拒绝覆盖。

完成训练后，验证全部必需产物、解析最终验证指标并计算模型文件的聚合 SHA-256：

```bash
python reproduction/audit_full_training.py
```

也可先用 `--family competitors` 或 `--family gcnn` 审计已经全部完成的单个模型族；结果会增量合并至 `reproduction/results/full_training_manifest.json`。

训练审计通过后，四类问题的官方 20,000-sample imitation-accuracy 测试可并行运行：

```bash
bash reproduction/run_full_test_queue.sh
```

它对每个问题直接调用 `04_test.py`，覆盖全部官方模型与 5 个 seed；结果和日志保存在 `reproduction_artifacts/full_test/`。

在线求解评测严格保留官方的 60 个迁移实例、全部策略、5 个 seed 和每次 3600 秒 CPU 时限。`05_evaluate.py` 的默认行为不变，但提供了可选过滤参数，使总计 6,600 个独立 SCIP 求解可以通过可恢复队列并行执行：

```bash
LEARN2BRANCH_EVALUATION_JOBS=80 bash reproduction/run_full_evaluation_queue.sh
```

每个分片固定使用一个 CPU，结果、日志和完成标记按时限隔离保存在 `reproduction_artifacts/full_evaluation/3600s/`。
