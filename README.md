# LSTMToPredictStock

基于 **LSTM 神经网络** 的贵州茅台（600519.SH）股票涨跌方向预测系统。

## 项目简介

使用长短时记忆网络（LSTM）对贵州茅台的日频交易数据进行建模，预测次一交易日的涨跌方向（二分类）。采用 **Expanding Window（扩展窗口）** 回测框架评估模型的泛化能力。

本项目可作为量化金融、深度学习在金融时间序列预测中的应用案例，适用于技术报告、论文或保研材料。

## 目录结构

```
├── src/
│   ├── train.py              # 训练主脚本（入门版）
│   ├── ModelOptimize.py      # 优化后的训练脚本
│   ├── visualize.py          # 可视化模块（论文级图表）
│   └── PrepareData.py        # 数据采集脚本（baostock）
├── diagrams/                 # 生成的图表输出目录
├── log/                      # 训练日志
├── data/                     # 数据文件
├── model/                    # 保存的模型权重
└── README.md
```

## 模型架构

```
输入 (batch, look_back=20, features=6)
  ↓
LSTM (2层, hidden_size=64, dropout=0.3)
  ↓
取最后一个时间步输出 (batch, 64)
  ↓
Linear(64→32) → ReLU → Dropout(0.2)
  ↓
Linear(32→2) → 类别 logits
  ↓
CrossEntropyLoss → 涨/跌 二分类
```

**输入特征（6个）**：
| 特征 | 说明 |
|------|------|
| `close` | 收盘价 |
| `volume` | 成交量 |
| `ma5` | 5日移动均线 |
| `ma20` | 20日移动均线 |
| `rsi14` | 14日相对强弱指标 |
| `vol_change` | 成交量变化率 |

## 训练方法

### Expanding Window（扩展窗口）回测

- 训练集：从 2015-01-01 到每个锚点日期（逐年扩展）
- 测试集：锚点日期后 6 个月
- 锚点：2018-12-31 至 2025-12-31，每年一个窗口
- 每个窗口独立初始化模型，从零开始训练
- 早停机制：验证准确率连续 15 轮不创新高则停止

## 快速开始

### 1. 安装依赖

```bash
pip install torch pandas numpy scikit-learn matplotlib baostock
# SHAP 特征重要性（可选）
pip install shap
```

### 2. 获取数据

```bash
python src/PrepareData.py
```

### 3. 训练模型

```bash
# 优化版
python src/ModelOptimize.py

# 入门版（含可视化）
python src/train.py
```

### 4. 查看图表

图表自动生成在 `diagrams/` 目录下：

| 图表 | 说明 |
|------|------|
| `loss_curve_{date}.png` | 训练/验证损失曲线（含早停标记） |
| `time_series_{date}.png` | 实际收益率 + 预测上涨概率时序图 |
| `confusion_matrix_{date}.png` | 混淆矩阵（含准确率、精确率、召回率、F1） |
| `cumulative_return_{date}.png` | 多空策略 vs 买入持有累积收益对比 |
| `shap_importance_{date}.png` | SHAP 特征重要性（需安装 shap） |
| `summary_dashboard.png` | 所有窗口性能汇总 |

## 实验结果

各窗口验证集最佳准确率（基于优化版，look_back=20）：

| 窗口 | 训练期 | 测试期 | 最佳验证准确率 |
|:----:|--------|--------|:-------------:|
| 1 | ≤2018-12-31 | 2019-01~06 | 52.5% |
| 2 | ≤2019-12-31 | 2020-01~06 | 43.9% |
| 3 | ≤2020-12-31 | 2021-01~06 | 50.5% |
| 4 | ≤2021-12-31 | 2022-01~06 | 54.1% |
| 5 | ≤2022-12-31 | 2023-01~06 | **59.2%** |
| 6 | ≤2023-12-31 | 2024-01~06 | **59.2%** |
| 7 | ≤2024-12-31 | 2025-01~06 | 57.1% |
| 8 | ≤2025-12-31 | 2026-01~06 | **63.9%** |

## 日志查看

训练日志保存在 `log/` 目录下，包含每轮损失、训练准确率和验证准确率。

```bash
# 查看最新日志
tail -50 log/LSTM_Optimize*.log
```

## 技术栈

- **框架**：PyTorch
- **数据处理**：Pandas, NumPy, Scikit-learn
- **可视化**：Matplotlib
- **数据源**：Baostock
- **特征分析**：SHAP（可选）
