import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from ModEval import evaluate_model
from DataHandler import data_handler
from xgboost import XGBClassifier
from ID_Finder import id_finder


# 读取数据
data = pd.read_excel('data.xlsx')
stage = 2
data = data_handler(data, stage)
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
education = education[['id', 'degree_raw', 'major_final', 'university_final', 'country_final']]
work = work[['id', 'title', 'orz', 'country_final']]
education = education.dropna()
work = work.dropna()
education = (pd.get_dummies(education, columns=education.columns.difference(['id'])).groupby('id').max().reset_index().astype(int))
data = pd.merge(data, education, on='id', how='outer')
data = data.dropna(subset=['interviewed'])
work = pd.get_dummies(work, columns=work.columns.difference(['id'])).groupby('id').max().reset_index().astype(int)
data = pd.merge(data, work, on='id', how='outer')
data = data.dropna(subset=['interviewed'])
# 将新特征重命名以避免特殊符号
original_columns = data.columns.tolist()
new_columns = original_columns[:28] + [f'feature_{i}' for i in range(1, len(original_columns) - 27)]
data.columns = new_columns

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
id_finder(data, 'XGBoost_WithDetail', stage)
# 1轮准确率为： 0.6850921273031826
# 1轮召回率为： 0.5246523388116309

# 2轮准确率为： 0.6583034647550776
# 2轮召回率为： 0.2393617021276596