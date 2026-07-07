import time
from copy import deepcopy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import datetime
import os

BATCH_SIZE=16
EPOCHS=100
PATIENCE=15

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

#构造LSTM监督学习样本
def create_supervised_data(X_raw,y_raw,look_back):
    X,y=[],[]
    for i in range(len(X_raw)-look_back):
        X.append(X_raw[i:i+look_back])
        y.append(y_raw[i+look_back])
    return np.array(X),np.array(y)

#数据预处理
def data_load(look_back):
    df=pd.read_csv('../data/stock_day_price.csv')
    df['date']=pd.to_datetime(df['date'])
    num_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
    df[num_cols]=df[num_cols].apply(pd.to_numeric,errors='coerce')

    df=df.sort_values('date').reset_index(drop=True)

    #收益率，[Pt/P(t-1)]-1
    df['ret']=df['close'].pct_change()
    df['label']=(df['ret'].shift(-1)>0).astype(int)   #标签，涨跌方向


    #构造特征
    df['ma5']=df['close'].rolling(5).mean()
    df['ma20']=df['close'].rolling(20).mean()

    #RSI相对强弱指标，越靠近100说明价格近期在涨(超买)，靠近0表示跌（超卖），50表示平衡
    delta=df['close'].diff()   #后一行减前一行,计算差值
    gain=delta.clip(lower=0).rolling(14).mean()
    loss =(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss
    df['rsi14']=100-100/(1+rs)

    #成交量变化率
    df['vol_change']=df['volume'].pct_change()

    df=df.dropna().reset_index(drop=True)

    #划分数据集(Expanding window)
    features = ['close', 'volume', 'ma5', 'ma20', 'rsi14', 'vol_change']
    target = 'label'

    anchors=pd.date_range(
        start='2018-12-31',
        end='2025-12-31',
        freq='Y'
    )
    datasets=[]

    for anchor in anchors:
        train_df=df[df['date']<=anchor]
        test_start = anchor + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=6)
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if(len(test_df))<look_back:
            continue

        #数据标准化
        transfer=StandardScaler()
        X_train=transfer.fit_transform(train_df[features])
        X_test=transfer.transform(test_df[features])
        y_train=train_df[target].values
        y_test=test_df[target].values

        # 构造监督样本
        X_train_seq, y_train_seq = create_supervised_data(
            X_train, y_train, look_back
        )
        X_test_seq, y_test_seq = create_supervised_data(
            X_test, y_test, look_back
        )

        train_dataset=TensorDataset(torch.tensor(X_train_seq,dtype=torch.float32),torch.tensor(y_train_seq,dtype=torch.long))
        test_dataset=TensorDataset(torch.tensor(X_test_seq,dtype=torch.float32),torch.tensor(y_test_seq,dtype=torch.long))
        datasets.append((train_dataset,test_dataset))
    return datasets

#构造LSTM神经网络模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=2, batch_first=True, dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

#创建早停类
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        metric = score if self.mode == 'max' else -score

        if self.best_score is None:
            self.best_score = metric
            self.best_state = deepcopy(model.state_dict())
        elif metric < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.best_state = deepcopy(model.state_dict())
            self.counter = 0

#模型训练和评估
def model_train_evaluate(input_size, logger, datasets):
    criterion=nn.CrossEntropyLoss()
    for train_dataset,test_dataset in datasets:
        model = LSTMModel(input_size)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=PATIENCE)
        train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False)
        test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
        for epoch in range(EPOCHS):
            model.train()
            start = time.time()
            # 定义变量，记录每次训练的损失值,批次数
            total_loss, batch_num,train_correct = 0.0, 0,0
            for x,y in train_loader:
                y_pred=model(x)
                loss=criterion(y_pred,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累计损失值
                total_loss += loss.item() * x.size(0)  # 把本轮的每批次(16条)的平均损失累积起来
                batch_num += x.size(0)
                train_correct += (y_pred.argmax(dim=-1) == y).sum().item()
            #测试
            model.eval()
            test_loss,correct=0.0,0
            with torch.no_grad():
                for x, y in test_loader:
                    y_pred = model(x)
                    # 根据加权求和得到类别，用argmax函数获取最大值对应的下标，就是类别
                    y_pred = torch.argmax(y_pred, dim=-1)  # dim=-1表示逐行处理
                    # 计算准确率
                    correct += (y_pred == y).sum().item()
            test_accuracy = correct / len(test_dataset)
            train_accuracy=train_correct / len(train_dataset)
            logger.info(
                f'当前轮数:{epoch}，当前轮的平均损失:{total_loss / batch_num:.4f},当前轮训练集的正确率(Accuracy):{train_accuracy * 100:.4f}%,当前轮测试集的正确率为:{test_accuracy * 100:.4f}%,耗时：{time.time() - start:.4f}s')
            print(
                f'当前轮数:{epoch}，当前轮的平均损失:{total_loss / batch_num:.4f},当前轮训练集的正确率(Accuracy):{train_accuracy * 100:.4f}%,当前轮测试集的正确率为:{test_accuracy * 100:.4f}%,耗时：{time.time() - start:.4f}s')
            # 早停策略评估
            early_stopping(test_accuracy, model)
            if early_stopping.early_stop:
                print(f"早停策略触发。当前训练轮数为:{epoch}")
                logger.info(f"早停策略触发。当前训练轮数为:{epoch}")
                break
        logger.info(f"模型训练结束,当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        torch.save(early_stopping.best_state, '../model/LSTM_optimization.pth')


if __name__ == '__main__':
    # 日志
    logfile_name = 'LSTM_Optimize' + datetime.datetime.now().strftime('%Y%m%d')
    logger = Logger('../', logfile_name).get_logger()
    datasets=data_load(40)
    _, look_back, input_size = datasets[0][0].tensors[0].shape
    logger.info(f"LSTM input_size={input_size}, hidden_size=64, layers=2")
    model_train_evaluate(input_size, logger, datasets)
