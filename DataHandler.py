import pandas as pd
import numpy as np

# 数据加载
data = pd.read_excel('data.xlsx')
# 根据面试信息创建ID
data['task_id'] = data.groupby(['session', 'mandate']).ngroup() + 1
# 筛选数据
columns_to_drop = [1, 3, 4, 6, 7, 12, 13, 14, 35]
data = data.drop(data.columns[columns_to_drop], axis=1)
# 生成是否进面试信息
data['interviewed'] = np.where(data['category'] == 'Eligible', 0, 1)
data = data.drop(columns='category')
# # 将所有数值列转换为整数
# data[data.select_dtypes(include=['float', 'int']).columns] = data.select_dtypes(include=['float', 'int']).astype(int)

data.to_csv('df.csv', index=False)