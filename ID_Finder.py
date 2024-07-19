from sklearn.metrics import precision_score
import pandas as pd


def id_finder(data, stage, model_type):
    # 创建空白列表
    result = []
    if stage == 1:
        # 提取每个task_id对应信息
        for task_id in data['task_id'].unique():
            test = data[data['task_id'] == task_id]
            # 获取对应ID
            actual_id = test.loc[test['interviewed'] == 1, 'id']
            actual_id = ';'.join(actual_id.astype(str))
            pred_id = test.loc[test['pred'] == 1, 'id']
            pred_id = ';'.join(pred_id.astype(str))
            # 计算精确率
            prec = precision_score(test['interviewed'], test['pred'])
            # 将结果添加到列表中
            result.append({'task_id': task_id, 'Actual_ID': actual_id,
                           'Predicted_ID': pred_id, 'Precision': prec})
    if stage == 2:
        # 提取每个task_id对应信息
        for task_id in data['task_id'].unique():
            test = data[data['task_id'] == task_id]
            # 获取对应ID
            actual_id = test.loc[test['selected'] == 1, 'id']
            actual_id = ';'.join(actual_id.astype(str))
            pred_id = test.loc[test['pred'] == 1, 'id']
            pred_id = ';'.join(pred_id.astype(str))
            # 计算精确率
            prec = precision_score(test['selected'], test['pred'])
            # 将结果添加到列表中
            result.append({'task_id': task_id, 'Actual_ID': actual_id,
                           'Predicted_ID': pred_id, 'Precision': prec})
    # 将结果转为数据集
    result = pd.DataFrame(result)
    if model_type == 'XGBoost':
        result.rename(columns={'Predicted_ID': 'Predicted_ID_XGBoost',
                               'Precision_ID': 'Precision_XGBoost'}, inplace=True)
    elif model_type == 'DeepSeek':
        result.rename(columns={'Predicted_ID': 'Predicted_ID_DeepSeek',
                               'Precision_ID': 'Precision_DeepSeek'}, inplace=True)
    return result