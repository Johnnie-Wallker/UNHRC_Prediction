import pandas as pd
from sklearn.metrics import accuracy_score
from ModEval import evaluate_model
from xgboost import XGBClassifier
from ID_Finder import id_finder


# 读取数据
data = pd.read_csv('df.csv')
# 将文本转化为类别
data['nationality_final'] = data['nationality_final'].astype('category')
data['mandate'] = data['mandate'].astype('category')
# 计算每个 task_id 的组内平均值
features_to_average = [col for col in data.columns if col not in ['id', 'mandate', 'gender_final', 'selected',
                                                                  'nationality_final', 'task_id', 'interviewed']]
grouped_df = data.groupby('task_id')[features_to_average].transform('mean')
# 生成新的个人-平均特征值
for feature in features_to_average:
    data[f'{feature}_diff'] = data[feature] - grouped_df[feature]
# 创建XGBoost
xgb = XGBClassifier(objective='binary:logistic', seed=1, enable_categorical=True)
# 将结果转为数据集
pred = pd.DataFrame(evaluate_model(xgb, data, stage=1))
# data = data[data['interviewed'] == 1]
data = pd.merge(data, pred, on='id', how='left')
# 计算准确率
acc = accuracy_score(data['interviewed'], data['pred'])
print('准确率为：', acc)
# 获取具体ID信息
result_xgb = id_finder(data, 1, 'XGBoost')
result_xgb.to_csv('result_XGBoost.csv', index=False)
# 1轮准确率为：准确率为： 0.6778964667214462
# 2轮准确率为：0.6630824372759857