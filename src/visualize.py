"""
可视化模块 — 分类版：上涨/下跌预测
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_academic_style():
    """设置学术风格的 matplotlib 参数（支持中文）"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def save_loss_curve(
    history: dict,
    anchor: pd.Timestamp,
    stopped_epoch: int,
    save_dir: str
):
    """训练 / 验证损失曲线（含早停标记）"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, history["train_loss"], label="训练损失",
            linewidth=1.2, color='#1f77b4')
    ax.plot(epochs, history["test_loss"], label="验证损失",
            linewidth=1.2, color='#d62728')
    ax.axvline(x=stopped_epoch, color='gray', linestyle='--', alpha=0.6,
               linewidth=1, label=f"早停 (第 {stopped_epoch} 轮)")

    ax.set_xlabel("训练轮数")
    ax.set_ylabel("交叉熵损失 (CrossEntropy)")
    ax.set_title(f"训练与验证损失曲线（训练截止: {anchor.date()}）")
    ax.legend()
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_curve_{anchor.date()}.png"))
    plt.close()


def save_time_series_plot(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_returns: np.ndarray,
    test_dates: np.ndarray,
    anchor: pd.Timestamp,
    accuracy: float,
    save_dir: str
):
    """分类版时序图：
       - 上：实际收益率柱状图（绿涨红跌）
       - 下：预测上涨概率曲线（含决策阈值 0.5）
    """
    device = next(model.parameters()).device
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # ====== 上：实际收益率柱状图 ======
    bar_colors = ['#d62728' if r < 0 else '#1f77b4' for r in y_returns]
    ax1.bar(test_dates, y_returns, color=bar_colors, width=1, alpha=0.7)
    ax1.axhline(y=0, color='gray', linewidth=0.6)
    ax1.set_ylabel("实际收益率")
    ax1.set_title(
        f"实际收益率与预测概率 — {anchor.date()}+6M  "
        f"(准确率 = {accuracy:.2%})"
    )
    ax1.set_axisbelow(True)

    # ====== 下：预测上涨概率 ======
    ax2.plot(test_dates, probs, label="预测上涨概率",
             linewidth=1.0, color='#d62728')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8,
                alpha=0.7, label="决策阈值 (0.5)")
    # 填充置信区域
    ax2.fill_between(test_dates, 0.5, probs, where=(probs >= 0.5),
                     alpha=0.12, color='#d62728')
    ax2.fill_between(test_dates, probs, 0.5, where=(probs < 0.5),
                     alpha=0.12, color='#1f77b4')

    # 在顶部用散点标注实际方向
    up_mask = y_test == 1
    down_mask = y_test == 0
    ax2.scatter(test_dates[up_mask], np.full(up_mask.sum(), 1.02),
                marker='.', s=6, color='#1f77b4', alpha=0.5, label="实际上涨")
    ax2.scatter(test_dates[down_mask], np.full(down_mask.sum(), -0.02),
                marker='.', s=6, color='#d62728', alpha=0.5, label="实际下跌")

    ax2.set_ylabel("预测上涨概率")
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_xlabel("日期")
    ax2.legend(ncol=3, fontsize=8)
    ax2.set_axisbelow(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"time_series_{anchor.date()}.png"))
    plt.close()

    return preds


def save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    anchor: pd.Timestamp,
    accuracy: float,
    save_dir: str
):
    """混淆矩阵"""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0, vmax=cm.max() * 1.3)

    # 色条
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 标注数字
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测下跌", "预测上涨"])
    ax.set_yticklabels(["实际下跌", "实际上涨"])
    ax.set_xlabel("预测类别")
    ax.set_ylabel("实际类别")
    ax.set_title(f"混淆矩阵 — {anchor.date()}+6M  (准确率 = {accuracy:.2%})")

    # 在底部添加统计指标
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    stats_text = (f"精确率 (Precision) = {precision:.2%}  "
                  f"召回率 (Recall) = {recall:.2%}  "
                  f"F1 = {f1:.3f}")
    ax.text(0.5, -0.18, stats_text, ha="center", va="center",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{anchor.date()}.png"))
    plt.close()


def save_cumulative_return(
    y_returns: np.ndarray,
    y_pred: np.ndarray,
    test_dates: np.ndarray,
    anchor: pd.Timestamp,
    save_dir: str
):
    """策略累积收益：多空策略 vs 买入持有

    y_returns: 实际收益率（连续值）, y_pred: 预测类别 (0/1)
    """
    # 预测上涨→做多(1)，预测下跌→做空(-1)
    position = np.where(y_pred == 1, 1, -1)
    strategy_returns = position * y_returns
    cum_strategy = np.cumprod(1 + strategy_returns) - 1
    cum_hold = np.cumprod(1 + y_returns) - 1

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(test_dates, cum_strategy * 100, alpha=0.08, color='#d62728')
    ax.plot(test_dates, cum_strategy * 100, label="多空策略 (预测方向)",
            linewidth=1.3, color='#d62728')
    ax.plot(test_dates, cum_hold * 100, label="买入持有",
            linewidth=1.3, color='#1f77b4', alpha=0.7)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益率 (%)")
    ax.set_title(f"策略回测 — {anchor.date()}+6M")
    ax.legend()
    ax.set_axisbelow(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cumulative_return_{anchor.date()}.png"))
    plt.close()


def save_shap_importance(
    model: torch.nn.Module,
    X_train_bg: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    anchor: pd.Timestamp,
    save_dir: str
) -> bool:
    """SHAP 特征重要性分析（需安装 shap 包）"""
    try:
        import shap
    except ImportError:
        return False

    device = next(model.parameters()).device
    model.eval()

    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)

    n_bg = min(50, len(X_train_bg))
    n_test = min(100, len(X_test))
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train_bg), n_bg, replace=False)
    test_idx = rng.choice(len(X_test), n_test, replace=False)

    background = torch.tensor(X_train_bg[bg_idx], dtype=torch.float32).to(device)
    test_samples = torch.tensor(X_test[test_idx], dtype=torch.float32).to(device)

    shap_values = None
    for cls in (shap.DeepExplainer, shap.GradientExplainer):
        try:
            explainer = cls(_Wrapper(model), background)
            sv = explainer.shap_values(test_samples)
            if isinstance(sv, list):
                sv = np.array(sv)
            break
        except Exception:
            sv = None

    if sv is None:
        return False

    # sv shape: (n_test, look_back, n_features, n_classes) or (n_test, look_back, n_features)
    if sv.ndim == 4:
        # Take SHAP values for class 1 (up)
        sv = sv[:, :, :, 1]
    importance = np.abs(sv).mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(importance)
    ax.barh(range(len(sorted_idx)), importance[sorted_idx],
            color='#1f77b4', edgecolor='white', height=0.6)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("平均 |SHAP| 值")
    ax.set_title(f"SHAP 特征重要性（上涨类别）— {anchor.date()}")
    ax.set_axisbelow(True)

    for i, idx in enumerate(sorted_idx):
        ax.text(importance[idx] + importance.max() * 0.01, i,
                f"{importance[idx]:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"shap_importance_{anchor.date()}.png"))
    plt.close()
    return True


def save_summary_dashboard(results_df: pd.DataFrame, save_dir: str):
    """汇总仪表盘：准确率柱状图 + 测试样本量"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [
        f"{s}\n→ {e}"
        for s, e in zip(results_df["test_start"], results_df["test_end"])
    ]
    colors = ["#d32f2f" if acc < 0.5 else "#388e3c" for acc in results_df["accuracy"]]

    # === 准确率柱状图 ===
    ax = axes[0]
    bars = ax.bar(range(len(results_df)), results_df["accuracy"],
                  color=colors, width=0.6, edgecolor='white')
    ax.axhline(y=0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("准确率 (Accuracy)")
    mean_acc = results_df["accuracy"].mean()
    beat_ratio = results_df["accuracy"].gt(0.5).mean() * 100
    ax.set_title(
        f"各窗口准确率  |  均值 = {mean_acc:.2%}  |  超50% = {beat_ratio:.0f}%",
        fontweight="bold"
    )
    for bar, acc in zip(bars, results_df["accuracy"]):
        offset = 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_axisbelow(True)

    # === 样本量柱状图 ===
    ax = axes[1]
    ax.bar(range(len(results_df)), results_df["n_samples"],
           color='#1976d2', width=0.6, edgecolor='white')
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("测试样本数")
    ax.set_title("各窗口测试样本量", fontweight="bold")
    for i, n in enumerate(results_df["n_samples"]):
        ax.text(i, n + max(results_df["n_samples"]) * 0.02,
                str(n), ha="center", fontsize=8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_dashboard.png"))
    plt.close()
