import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from ModEval import evaluate_model
from DataHandler import data_handler
from xgboost import XGBClassifier
from ID_Finder import id_finder


# 读取数据
data = pd.read_excel('data.xlsx')
data = data_handler(data, stage=1)
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
education = education[['id', 'degree_raw', 'major_final', 'university_final', 'country_final']]
work = work[['id', 'title', 'orz', 'country_final']]
education = education.dropna()
work = work.dropna()
education = (pd.get_dummies(education, columns=education.columns.difference(['id'])).groupby('id').max().reset_index().astype(int))
df = pd.merge(data, education, on='id', how='outer')
df = df.dropna(subset=['interviewed'])
work = pd.get_dummies(work, columns=work.columns.difference(['id'])).groupby('id').max().reset_index().astype(int)
df = pd.merge(data, work, on='id', how='outer')
df = df.dropna(subset=['interviewed'])

# 创建XGBoost
xgb = XGBClassifier(objective='binary:logistic', seed=1, enable_categorical=True)
# 获取预测结果
pred = pd.DataFrame(evaluate_model(xgb, df))
df = pd.merge(df, pred, on='id', how='left')
# 计算准确率
acc = accuracy_score(df['interviewed'], df['pred'])
f1 = f1_score(df['interviewed'], df['pred'])
print('准确率为：', acc)
print('召回率为：', f1)
# 获取具体ID信息
id_finder(df, 'XGBoost_WithDetail')
# 准确率为： 0.6850921273031826
# 召回率为： 0.5246523388116309