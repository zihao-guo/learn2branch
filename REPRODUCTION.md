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
