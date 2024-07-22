import pandas as pd

# x = pd.read_csv('result_XGBoost.csv')
# y = pd.read_csv('result_DeepSeek.csv')
# z = pd.read_csv('result_DeepSeek_Summary.csv')
#
# q = pd.merge(x, y, on=['task_id', 'Actual_ID'], how='outer')
# p = pd.merge(q, z, on=['task_id', 'Actual_ID'], how='outer')
# p.to_csv('result.csv', index=False)
result = pd.read_csv('result.csv')
data = pd.read_csv('df.csv')
df = data.groupby('task_id')['mandate'].first().reset_index()
result = pd.merge(df, result, on='task_id', how='outer')
result.to_csv('result.csv', index=False)
