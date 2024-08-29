import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from Model_Evaluate import modeval
from Data_Handler import data_handler
from Result_Logger import log_result


def run_ml(**kwargs):
    data = pd.read_excel('data.xlsx')
    data = data_handler(data, kwargs['stage'])
    data['other nationality_final'] = data['other nationality_final'].astype('category')
    education = pd.read_excel('data.xlsx', sheet_name=1)
    work = pd.read_excel('data.xlsx', sheet_name=2)
    education = education[['id', 'degree_raw', 'major_final', 'university_final', 'country_final']]
    work = work[['id', 'title', 'orz', 'country_final']]
    education = education.dropna()
    work = work.dropna()
    if kwargs['detail']:
        if kwargs['detail'] != 'Work':
            education = pd.get_dummies(education,
                                       columns=education.columns.difference(['id'])).groupby(
                'id').max().reset_index().astype(int)
            data = pd.merge(data, education, on='id', how='outer')
            data = data.dropna(subset=['interviewed'])
        if kwargs['detail'] != 'Education':
            work = pd.get_dummies(work, columns=work.columns.difference(['id'])).groupby(
                'id').max().reset_index().astype(int)
            data = pd.merge(data, work, on='id', how='outer')
            data = data.dropna(subset=['interviewed'])
        original_columns = data.columns.tolist()
        new_columns = original_columns[:29] + [f'feature_{i}' for i in range(1, len(original_columns) - 28)]
        data.columns = new_columns
    model = kwargs['model']
    pred = pd.DataFrame(modeval(model, data))
    data = pd.merge(data, pred, on='id', how='left')
    acc = accuracy_score(data['interviewed'], data['pred'])
    f1 = f1_score(data['interviewed'], data['pred'])
    if kwargs['save_result']:
        suffix = ''
        if kwargs['detail']:
            if kwargs['detail'] == 'Education' or 'Work':
                suffix += f'_{kwargs["detail"]}'
            else:
                suffix += f'_FullDetail'
        log_result(data, kwargs['stage'], f'XGBoost_{suffix}')

    return {'accuracy': acc, 'f1_score': f1}