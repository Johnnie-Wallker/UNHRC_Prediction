import pandas as pd
pd.options.mode.copy_on_write = True


def modeval(model, data):
    # 创建空白列表
    result = []
    for i in range(len(data['task_id'].unique())):
        task_id = data['task_id'].unique()[i]
        train = data[data['task_id'] != task_id]
        test = data[data['task_id'] == task_id]
        # 划分训练测试集
        X_train = train.drop(['task_id', 'interviewed', 'id', 'count'], axis=1)
        y_train = train['interviewed']
        X_test = test.drop(['task_id', 'interviewed', 'id', 'count'], axis=1)
        test = test.copy()
        # 模型训练
        model.fit(X_train, y_train)
        # 获得每行预测为1的概率
        test.loc[:, 'probs'] = model.predict_proba(X_test)[:, 1]
        # 对组内的概率进行排序
        test.loc[:, 'rank'] = test['probs'].rank(method='first', ascending=False)
        # 将小于等于通过人数的行预测为1
        test.loc[:, 'pred'] = test.apply(lambda x: 1 if x['rank'] <= x['count'] else 0, axis=1)
        # 提取结果
        for j in range(test.shape[0]):
            result.append({'id': test['id'].tolist()[j], 'pred': test['pred'].tolist()[j]})
        print(f'Current Progress: {round((i/len(data["task_id"].unique()))*100,1)}%')

    return result