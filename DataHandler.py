import numpy as np


def data_handler(data, stage):
    # 根据面试信息创建ID
    data['task_id'] = data.groupby(['session', 'mandate']).ngroup() + 1
    # 筛选数据
    columns_to_drop = [1, 3, 4, 6, 7, 12, 13, 14, 35]
    data = data.drop(data.columns[columns_to_drop], axis=1)
    # 生成是否进面试信息
    data['interviewed'] = np.where(data['category'] == 'Eligible', 0, 1)
    data = data.drop(columns='category')
    # 将文本转化为类别
    data['nationality_final'] = data['nationality_final'].astype('category')
    data['mandate'] = data['mandate'].astype('category')
    if stage == 1:
        data['count'] = data.groupby('task_id')['interviewed'].transform('sum')
        data = data.groupby('task_id').filter(lambda x: len(x) != x['count'].iloc[0])
        data = data.drop(columns='selected')
    else:
        data = data[data['interviewed'] == 1]
        data['count'] = 1
        data = data.drop(columns='interviewed')
        data = data.rename(columns={'selected': 'interviewed'})

    return data