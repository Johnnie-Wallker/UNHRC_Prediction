import numpy as np
import hashlib


def int_md5_transform(num=None, md5_hash=None, length=6, reverse=False):
    if not hasattr(int_md5_transform, "md5_to_int_map"):
        int_md5_transform.md5_to_int_map = {}
        int_md5_transform.int_to_md5_map = {}
        for i in range(1, 3001):
            num_str = str(i)
            md5_hash = hashlib.md5(num_str.encode()).hexdigest()[:length]
            int_md5_transform.md5_to_int_map[md5_hash] = i
            int_md5_transform.int_to_md5_map[i] = md5_hash
    if reverse:
        if md5_hash in int_md5_transform.md5_to_int_map:
            return int_md5_transform.md5_to_int_map[md5_hash]
        else:
            raise ValueError("MD5 hash not found in the mapping.")
    else:
        if num in int_md5_transform.int_to_md5_map:
            return int_md5_transform.int_to_md5_map[num]
        else:
            raise ValueError("Integer not found in the mapping.")


def data_handler(data, stage):
    # 根据面试信息创建ID
    data['task_id'] = data.groupby(['session', 'mandate']).ngroup() + 1
    # 筛选数据
    columns_to_drop = [1, 3, 4, 7, 12, 13, 14, 35]
    data = data.drop(data.columns[columns_to_drop], axis=1)
    # 生成是否进面试信息
    data['interviewed'] = np.where(data['category'] == 'Eligible', 0, 1)
    data = data.drop(columns='category')
    if stage == 1:
        data['count'] = data.groupby('task_id')['interviewed'].transform('sum')
        data = data.groupby('task_id').filter(lambda x: len(x) != x['count'].iloc[0])
        data = data.drop(columns='selected')
    else:
        data = data[data['interviewed'] == 1]
        data = data.drop(columns='interviewed')
        data['count'] = data.groupby('task_id')['selected'].transform('sum')
        data = data[data['count'] != 0]
        data = data.rename(columns={'selected': 'interviewed'})
    # 将文本转化为类别
    data['nationality_final'] = data['nationality_final'].astype('category')
    data['mandate'] = data['mandate'].astype('category')

    return data