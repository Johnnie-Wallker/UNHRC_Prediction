from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def id_finder(data, model_type, stage):
    # 创建空白列表
    result = []
    # 提取每个task_id对应信息
    for task_id in data['task_id'].unique():
        test = data[data['task_id'] == task_id]
        # 获取对应ID
        actual_id = test.loc[test['interviewed'] == 1, 'id']
        actual_id = ';'.join(actual_id.astype(str))
        pred_id = test.loc[test['pred'] == 1, 'id']
        pred_id = ';'.join(pred_id.astype(str))
        # 计算精确率
        acc = accuracy_score(test['interviewed'], test['pred'])
        f1 = f1_score(test['interviewed'], test['pred'])
        # 将结果添加到列表中
        result.append({'task_id': task_id, 'Actual_ID': actual_id,
                       'Predicted_ID': pred_id, 'Accuracy': acc, 'Recall': f1})
    # 将结果转为数据集
    result = pd.DataFrame(result)
    result.rename(columns={'Predicted_ID': f'Predicted_ID_{model_type}',
                           'Accuracy': f'Accuracy_{model_type}',
                           'Recall': f'Recall_{model_type}'},
                  inplace=True)
    result.to_csv(f'Result_Round{stage}/result_{model_type}.csv', index=False)
