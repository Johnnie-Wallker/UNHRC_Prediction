import pandas as pd
pd.options.mode.copy_on_write = True


def evaluate_model(model, data, stage):
    # 创建空白列表
    result = []
    for task_id in data['task_id'].unique():
        train = data[data['task_id'] != task_id]
        test = data[data['task_id'] == task_id]
        if stage == 1:
            # 划分训练测试集
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
            # 提取结果
            for i in range(test.shape[0]):
                result.append({'id': test['id'].tolist()[i], 'pred': test['pred'].tolist()[i]})
        if stage == 2:
            train = train[train['interviewed'] == 1]
            test = test[test['interviewed'] == 1]
            # 划分训练测试集
            X_train = train.drop(['task_id', 'interviewed', 'selected', 'id'], axis=1)
            y_train = train['selected']
            X_test = test.drop(['task_id', 'interviewed', 'selected', 'id'], axis=1)
            test = test.copy()
            # 模型训练
            model.fit(X_train, y_train)
            # 获得每行预测为1的概率
            test.loc[:, 'probs'] = model.predict_proba(X_test)[:, 1]
            # 将概率最大值预测为1
            test['pred'] = 0
            test.loc[test['probs'].idxmax(), 'pred'] = 1
            # 提取结果
            for i in range(test.shape[0]):
                result.append({'id': test['id'].tolist()[i], 'pred': test['pred'].tolist()[i]})
    return result