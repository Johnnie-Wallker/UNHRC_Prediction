import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from ModEval import evaluate_model
from DataHandler import data_handler
from xgboost import XGBClassifier
from ID_Finder import id_finder


# 读取数据
data = pd.read_excel('data.xlsx')
stage = 1
data = data_handler(data, stage)
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
pred = pd.DataFrame(evaluate_model(xgb, data))
data = pd.merge(data, pred, on='id', how='left')
# 计算准确率
acc = accuracy_score(data['interviewed'], data['pred'])
f1 = f1_score(data['interviewed'], data['pred'])
print('准确率为：', acc)
print('召回率为：', f1)
# 获取具体ID信息
id_finder(data, 'XGBoost', stage)
# 简历筛选轮：
# 准确率为： 0.6758793969849246 召回率为： 0.5107458912768648

# 面试轮：
# 准确率为： 0.6795180722891566 召回率为： 0.2887700534759358
