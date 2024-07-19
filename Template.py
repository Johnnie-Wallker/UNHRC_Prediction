import pandas as pd
from xgboost import XGBClassifier

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
X = data.drop(['task_id', 'interviewed', 'selected', 'id'], axis=1)
y = data['interviewed']
xgb.fit(X, y)