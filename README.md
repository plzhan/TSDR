# TSDR

Temporal Smoothness Doubly Robust Learning for Debiased Knowledge Tracing

知识追踪任务通常只使用学生已经作答或平台已经记录的交互数据训练模型。但在真实自适应学习系统中，题目推荐、学生跳题、学生自主选择等机制会导致数据并非随机缺失，而是 MNAR, Missing Not At Random。这会让模型把有偏观测误认为真实掌握状态，进而影响后续推荐。

TSDR 的目标是：

- 用 Doubly Robust 学习修正 KT 中由非随机观测带来的选择偏差。
- 同时建模 propensity, 即交互被观测到的概率，以及 imputation, 即未观测交互的潜在误差。
- 在序列 KT 场景中加入 temporal smoothness 约束，缓解 DR 估计器的方差累积和训练震荡。
- 作为 plug-and-play 框架增强 DKT, AKT, simpleKT, FoLiBiKT, SparseKT, DisKT 等骨干模型。

## 文档结构

```text
TSDR/
  README.md
  main.py
  train.py
  data_loaders.py
  preprocess_data.py
  configs/
    example.yaml
  models/
    __init__.py
    akt.py
    diskt.py
    dkt.py
    drkt.py
    folibikt.py
    simplekt.py
    sparsekt.py
  utils/
    __init__.py
    augment_seq.py
    config.py
    file_io.py
    utils.py
    visualizer.py
```

## 核心思想

TSDR 将 KT 预测风险从只在观测日志上计算，扩展为面向完整潜在交互空间的偏差校正估计。它包含三类模型：

- KT predictor: 根据历史交互预测下一题答对概率。
- Propensity model: 估计某个 concept 或 item 在当前状态下被观测的概率。
- Imputation model: 估计反事实交互上的预测误差。

普通 DR 可以在 propensity 或 imputation 任一模型准确时保持无偏，但在序列学习中可能因为权重和误差噪声造成高方差。TSDR 因此对隐状态轨迹加入 temporal smoothness regularization，使知识状态变化更连续，降低训练不稳定性。

## 快速使用参考

在参考代码工程中运行普通 KT：

```powershell
cd TSDR
python main.py --model_name akt --data_name prob
```

启用 TSDR 训练：

```powershell
python main.py --model_name akt --data_name prob --dr --lambda 0.3
```

只启用 IPW：

```powershell
python main.py --model_name akt --data_name prob --ipw
```

只启用 imputation：

```powershell
python main.py --model_name akt --data_name prob --imput
```

## 致谢

本项目的部分代码基于 [DisKT](https://github.com/zyy-2001/DisKT) 进行修改与扩展，在此对原项目作者的开源贡献表示感谢。
