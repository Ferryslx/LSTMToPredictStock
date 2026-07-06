import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import datetime
import os

from utils.log import Logger
from visualize import (
    set_academic_style,
    save_loss_curve,
    save_time_series_plot,
    save_confusion_matrix,
    save_cumulative_return,
    save_shap_importance,
    save_summary_dashboard,
)

warnings.filterwarnings("ignore")


# 1. 数据加载 & 特征工程
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["date"])
    num_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")  #将字符串转成数值型，转不了的变成Nan

    df = df.sort_values("date").reset_index(drop=True)

    # 收益率
    df["ret"] = df["close"].pct_change()
    df["label"] = (df["ret"].shift(-1) > 0).astype(int)

    # 技术指标
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    #RSI相对强弱指标，越靠近100说明价格近期在涨(超买)，靠近0表示跌（超卖），50表示平衡
    delta = df["close"].diff()   #后一行减前一行，计算差值
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - 100 / (1 + rs)

    #成交量变化率
    df["vol_chg"] = df["volume"].pct_change()

    df = df.dropna().reset_index(drop=True)
    return df


# 2. 将时间序列转换为 LSTM 可用的监督学习样本
def create_supervised_data(X_raw, y_raw, look_back):
    X, y = [], []
    for i in range(len(X_raw) - look_back):
        X.append(X_raw[i:i + look_back])
        y.append(y_raw[i + look_back])
    return np.array(X), np.array(y)


# 3. Dataset
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 4. LSTM 模型（类封装）
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )
        self.fc = nn.Linear(hidden_size, 2)
        # 日志
        logfile_name = 'LSTM' + datetime.datetime.now().strftime('%Y%m%d')
        self.logger = Logger('../', logfile_name).get_logger()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 5. 模型训练及测试
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    logger = model.logger

    best_test_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    history = {"train_loss": [], "test_loss": []}
    stopped_epoch = epochs

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 测试
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                test_loss += criterion(preds, y_batch).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)

        # 早停检查
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"当前训练轮数 {epoch+1:02d} | "
                f"训练集损失: {avg_train_loss:.6f} | "
                f"测试集损失: {avg_test_loss:.6f}"
            )
            logger.info(
                f"当前训练轮数 {epoch+1:02d} | "
                f"训练集损失: {avg_train_loss:.6f} | "
                f"测试集损失: {avg_test_loss:.6f}"
            )

        if no_improve_epochs >= patience:
            stopped_epoch = epoch + 1
            print(f"早停: 第 {stopped_epoch} 轮停止，最佳测试损失: {best_test_loss:.6f}")
            logger.info(f"早停: 第 {stopped_epoch} 轮停止，最佳测试损失: {best_test_loss:.6f}")
            break

    # 恢复最佳模型权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, stopped_epoch


# 6. 模型评估（分类准确率）
def evaluate_classification(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device
):
    model.eval()
    model.to(device)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()

    accuracy = (preds == y_test).mean()
    return accuracy, preds

# 7. 拓展窗口处理主流程
def expanding_window_run(
    csv_path: str,
    look_back: int = 20
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_and_prepare_data(csv_path)

    features = ["close", "volume", "ma5", "ma20", "rsi14", "vol_chg"]
    target = "label"

    # 扩展窗口锚点
    test_anchors = pd.date_range(
        start="2018-12-31",
        end="2025-12-31",
        freq="Y"
    )

    results = []
    diagram_dir = "../diagrams"
    os.makedirs(diagram_dir, exist_ok=True)
    set_academic_style()

    for anchor in test_anchors:
        print(f"\n===== Train ≤ {anchor.date()} =====")

        train_df = df[df["date"] <= anchor]
        test_start = anchor + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=6)
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if len(test_df) < look_back:
            continue

        # ===== 标准化（仅标准化特征，不标准化标签）=====
        scaler = StandardScaler()
        X_train_raw = scaler.fit_transform(train_df[features])
        X_test_raw = scaler.transform(test_df[features])
        y_train_raw = train_df[target].values
        y_test_raw = test_df[target].values

        X_train, y_train = create_supervised_data(X_train_raw, y_train_raw, look_back)
        X_test, y_test = create_supervised_data(X_test_raw, y_test_raw, look_back)

        train_loader = DataLoader(
            StockDataset(X_train, y_train),
            batch_size=64,
            shuffle=False
        )
        test_loader = DataLoader(
            StockDataset(X_test, y_test),
            batch_size=64,
            shuffle=False
        )

        # ===== 模型 =====
        model = LSTMModel(
            input_size=len(features),
            hidden_size=64
        )
        logger=model.logger

        history, stopped_epoch = train_model(
            model,
            train_loader,
            test_loader,
            device,
            epochs=100
        )

        accuracy, y_pred = evaluate_classification(model, X_test, y_test, device)
        print(f"准确率 = {accuracy:.2%}")
        logger.info(f"准确率 = {accuracy:.2%}")

        # ===== 可视化（仅最后一个窗口保存图表）=====
        is_last_window = (anchor == test_anchors[-1])
        if is_last_window:
            test_dates = test_df["date"].iloc[look_back:].values
            y_returns = test_df["ret"].iloc[look_back:].values

            save_loss_curve(history, anchor, stopped_epoch, diagram_dir)
            save_time_series_plot(
                model, X_test, y_test, y_returns, test_dates,
                anchor, accuracy, diagram_dir
            )
            save_confusion_matrix(y_test, y_pred, anchor, accuracy, diagram_dir)
            save_cumulative_return(y_returns, y_pred, test_dates, anchor, diagram_dir)
            save_shap_importance(
                model, X_train, X_test, features, anchor, diagram_dir
            )

        results.append({
            "test_start": test_start.date(),
            "test_end": test_end.date(),
            "accuracy": accuracy,
            "n_samples": len(y_test)
        })

    results_df = pd.DataFrame(results)
    save_summary_dashboard(results_df, diagram_dir)
    return results_df


#测试
if __name__ == "__main__":
    result_df = expanding_window_run(
        csv_path="../data/stock_day_price.csv",
        look_back=20
    )

    print("\n最终结果为：")
    print(result_df)
    print(f"\n平均准确率: {result_df['accuracy'].mean():.2%}")
    print(f"准确率 > 50% 比例: {(result_df['accuracy'] > 0.5).mean()*100:.1f}%")