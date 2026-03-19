import pandas as pd
import numpy as np
import scipy.stats as st

# 1. 读取下载的CSV文件
df = pd.read_csv("tts_evaluation_results.csv")

# 2. 定义计算均值和95%置信区间的函数
def calculate_metrics(group):
    # 计算均值
    mean_mos = group['MOS'].mean()
    mean_cmos = group['C_MOS'].mean()
    
    # 计算 95% 置信区间 (CI)
    ci_mos = st.t.interval(0.95, len(group['MOS'])-1, loc=mean_mos, scale=st.sem(group['MOS']))
    ci_cmos = st.t.interval(0.95, len(group['C_MOS'])-1, loc=mean_cmos, scale=st.sem(group['C_MOS']))
    
    # 处理方差为0的情况（所有打分一样）
    mos_err = (ci_mos[1] - mean_mos) if not np.isnan(ci_mos[1]) else 0
    cmos_err = (ci_cmos[1] - mean_cmos) if not np.isnan(ci_cmos[1]) else 0
    
    return pd.Series({
        'MOS Mean': f"{mean_mos:.2f} ± {mos_err:.2f}",
        'C-MOS Mean': f"{mean_cmos:.2f} ± {cmos_err:.2f}"
    })

# 3. 按模型分组计算结果
result_summary = df.groupby('Model_Name').apply(calculate_metrics)

print("==== 评测结果 ====")
print(result_summary.to_string())