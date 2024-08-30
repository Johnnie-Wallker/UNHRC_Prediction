from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def log_result(data, stage, model_type, prompt_type=None, token_count=None):
    result = []
    for task_id in data['task_id'].unique():
        test = data[data['task_id'] == task_id]
        count = test['count'].max()
        size = len(test)
        actual_id = test.loc[test['interviewed'] == 1, 'id']
        actual_id = ';'.join(actual_id.astype(str))
        pred_id = test.loc[test['pred'] == 1, 'id']
        pred_id = ';'.join(pred_id.astype(str))
        acc = accuracy_score(test['interviewed'], test['pred'])
        f1 = f1_score(test['interviewed'], test['pred'])
        result.append({'task_id': task_id, 'Actual_ID': actual_id, 'Group_Size': size, 'Selected': count,
                       'Predicted_ID': pred_id, 'Accuracy': acc, 'Recall': f1})
    result = pd.DataFrame(result)
    result.rename(columns={'Predicted_ID': f'Predicted_ID_{model_type}',
                           'Accuracy': f'Accuracy_{model_type}',
                           'Recall': f'Recall_{model_type}'},
                  inplace=True)
    if model_type == 'DeepSeek':
        result['token_count'] = token_count
        file_dir = f'Result_Round{stage}/result_{model_type}_{prompt_type}.csv'
        result.to_csv(file_dir, index=False)
    else:
        file_dir = f'Result_Round{stage}/result_{model_type}.csv'
        result.to_csv(file_dir, index=False)

    print(f'Result saved to {file_dir}')
