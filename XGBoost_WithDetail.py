import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from Model_Evaluate import modeval
from Data_Handler import data_handler
from xgboost import XGBClassifier
from Result_Logger import log_result


# 读取数据
data = pd.read_excel('data.xlsx')
stage = 1
data = data_handler(data, stage)
data['other nationality_final'] = data['other nationality_final'].astype('category')
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
education = education[['id', 'degree_raw', 'major_final', 'university_final', 'country_final']]
work = work[['id', 'title', 'orz', 'country_final']]
education = education.dropna()
work = work.dropna()
education = pd.get_dummies(education,
                           columns=education.columns.difference(['id'])).groupby('id').max().reset_index().astype(int)
data = pd.merge(data, education, on='id', how='outer')
data = data.dropna(subset=['interviewed'])
work = pd.get_dummies(work, columns=work.columns.difference(['id'])).groupby('id').max().reset_index().astype(int)
data = pd.merge(data, work, on='id', how='outer')
data = data.dropna(subset=['interviewed'])
# 将新特征重命名以避免特殊符号
original_columns = data.columns.tolist()
new_columns = original_columns[:29] + [f'feature_{i}' for i in range(1, len(original_columns) - 28)]
data.columns = new_columns

# 创建XGBoost
xgb = XGBClassifier(objective='binary:logistic', seed=1, enable_categorical=True)
# 获取预测结果
pred = pd.DataFrame(modeval(xgb, data))
data = pd.merge(data, pred, on='id', how='left')
# 计算准确率
acc = accuracy_score(data['interviewed'], data['pred'])
f1 = f1_score(data['interviewed'], data['pred'])
print(f'准确率为：{round(acc, 3)} 召回率为：{round(f1, 3)}')
# 获取具体ID信息
log_result(data, stage, 'XGBoost_WithDetail')
# 简历筛选轮：
# 准确率为：0.698 召回率为：0.544

# 面试轮：
# 准确率为：0.680 召回率为：0.289
