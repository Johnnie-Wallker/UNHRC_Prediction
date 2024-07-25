import pandas as pd

x = pd.read_csv(f'Result_Round1/result_XGBoost_WithDetail.csv')
y = pd.read_csv(f'Result_Round1/result_DeepSeek_Train.csv')
q = pd.merge(x, y, on=['task_id', 'Actual_ID'], how='outer')
q = q.drop(['Accuracy_XGBoost_WithDetail', 'Accuracy_DeepSeek_Train'], axis=1)
q['diff'] = q['Recall_XGBoost_WithDetail'] - q['Recall_DeepSeek_Train']
