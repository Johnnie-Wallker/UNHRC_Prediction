import pandas as pd
pd.options.mode.copy_on_write = True


def modeval(model, data):
    result = []
    for i in range(len(data['task_id'].unique())):
        task_id = data['task_id'].unique()[i]
        train = data[data['task_id'] != task_id]
        test = data[data['task_id'] == task_id]
        X_train = train.drop(['task_id', 'interviewed', 'id', 'count'], axis=1)
        y_train = train['interviewed']
        X_test = test.drop(['task_id', 'interviewed', 'id', 'count'], axis=1)
        test = test.copy()
        model.fit(X_train, y_train)
        test.loc[:, 'probs'] = model.predict_proba(X_test)[:, 1]
        test.loc[:, 'rank'] = test['probs'].rank(method='first', ascending=False)
        test.loc[:, 'pred'] = test.apply(lambda x: 1 if x['rank'] <= x['count'] else 0, axis=1)
        for j in range(test.shape[0]):
            result.append({'id': test['id'].tolist()[j], 'pred': test['pred'].tolist()[j]})
        print(f'Current Progress: {round((i/len(data["task_id"].unique()))*100,1)}%')

    return result