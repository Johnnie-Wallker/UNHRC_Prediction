import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from itertools import product
pd.options.mode.copy_on_write = True


def evaluate_model(model, data):
    # 划分训练测试集
    train = data[data['task_id'] <= data['task_id'].max()*0.8]
    test = data[data['task_id'] > data['task_id'].max()*0.8]
    X_train = train.drop(['task_id', 'interviewed', 'selected', 'id'], axis=1)
    y_train = train['interviewed']
    X_test = test.drop(['task_id', 'interviewed', 'selected', 'id'], axis=1)
    test = test.copy()
    # 模型训练
    model.fit(X_train, y_train)
    # 获得每行预测为1的概率
    test.loc[:, 'probs'] = model.predict_proba(X_test)[:, 1]
    # 计算面试通过人数
    test.loc[:, 'count'] = test['interviewed'].sum()
    # 对组内的概率进行排序
    test.loc[:, 'rank'] = test['probs'].rank(method='first', ascending=False)
    # 将小于等于通过人数的行预测为1
    test.loc[:, 'pred'] = test.apply(lambda x: 1 if x['rank'] <= x['count'] else 0, axis=1)
    # 计算准确率
    acc = accuracy_score(test['interviewed'], test['pred'])

    return acc


# 读取数据
data = pd.read_csv('df.csv')
# 将文本转化为类别
data['nationality_final'] = data['nationality_final'].astype('category')
data['mandate'] = data['mandate'].astype('category')
# 定义XGBoost参数空间
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [1, 3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 创建所有参数组合
param_combinations = list(product(*param_grid.values()))

# 初始化最佳参数和最佳准确率
best_params = None
best_accuracy = 0

# 遍历每个参数组合
for params in param_combinations:
    xgb = XGBClassifier(objective='binary:logistic', seed=1,
                        enable_categorical=True, **dict(zip(param_grid.keys(), params)))
    accuracy = evaluate_model(xgb, data)

    # 更新最佳参数和最佳准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print("Best Accuracy:", best_accuracy)
print("Best Parameters:", best_params)