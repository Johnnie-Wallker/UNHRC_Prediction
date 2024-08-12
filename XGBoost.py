import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from Model_Evaluate import modeval
from Data_Handler import data_handler
from xgboost import XGBClassifier
from Result_Logger import log_result


# 读取数据
data = pd.read_excel('data.xlsx')
stage = 2
data = data_handler(data, stage)
data['other nationality_final'] = data['other nationality_final'].astype('category')
# # 计算每个 task_id 的组内平均值
# features_to_average = [col for col in data.columns if col not in ['id', 'mandate', 'gender_final', 'selected',
#                                                                   'nationality_final', 'task_id', 'interviewed']]
# grouped_df = data.groupby('task_id')[features_to_average].transform('mean')
# # 生成新的个人-平均特征值
# for feature in features_to_average:
#     data[f'{feature}_diff'] = data[feature] - grouped_df[feature]
# 创建XGBoost
xgb = XGBClassifier(objective='binary:logistic', seed=1, enable_categorical=True)
# 获取预测结果
pred = pd.DataFrame(modeval(xgb, data))
data = pd.merge(data, pred, on='id', how='left')
# 计算准确率
acc = accuracy_score(data['interviewed'], data['pred'])
f1 = f1_score(data['interviewed'], data['pred'])
print('准确率为：', acc, '召回率为：', f1)
# 获取具体ID信息
log_result(data, 'XGBoost', stage)
# 简历筛选轮：
# 准确率为： 0.6825795644891123 召回率为： 0.5208596713021492

# 面试轮：
# 准确率为： 0.689156626506024 召回率为： 0.31016042780748665
