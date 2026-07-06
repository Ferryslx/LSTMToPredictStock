import baostock as bs
import pandas as pd

# ========== 配置 ==========
stock_code = "sh.600519"          # 贵州茅台，深交所用 sz.开头
start_date = "2015-01-01"
end_date   = "2026-07-01"
adjustflag = "2"                  # 2=前复权，3=不复权，1=后复权
fields     = "date,code,open,high,low,close,volume,amount,turn,pctChg,adjustflag"


# ========== 登录 ==========
lg = bs.login()
print(f"登录: error_code={lg.error_code}, error_msg={lg.error_msg}")

# ---------- 获取日K线 ----------
rs_day = bs.query_history_k_data_plus(
    stock_code,
    fields,
    start_date=start_date,
    end_date=end_date,
    frequency="d",       # d=日K线
    adjustflag=adjustflag
)

data_list = []
while rs_day.error_code == '0' and rs_day.next():
    data_list.append(rs_day.get_row_data())

df_day = pd.DataFrame(data_list, columns=rs_day.fields)
df_day.to_csv("../data/stock_day_price.csv", index=False, encoding="utf-8-sig")
print(f"日K线已保存，共 {len(df_day)} 条 → data/stock_day_price.csv")

# ========== 登出 ==========
bs.logout()