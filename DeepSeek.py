import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score
from Prompt_Generator import prompt_generator
from Data_Handler import data_handler, int_md5_transform
from Result_Logger import log_result
from collections import Counter


def run_deepseek(config, save_result=False):
    # 参数读取
    prompt_type = config['prompt_type']
    stage = config['stage']
    shuffle = config['shuffle']
    detail = config['detail']
    vote_count = config['vote_count']
    client = config['client']
    model = config['model']
    # 读取数据
    data = pd.read_excel('data.xlsx')
    data = data_handler(data, stage)
    education = pd.read_excel('data.xlsx', sheet_name=1)
    work = pd.read_excel('data.xlsx', sheet_name=2)
    # 填充空缺值
    data = data.fillna(0)
    data.iloc[:, 5:25] = data.iloc[:, 5:25].astype('int')
    # 替换年龄空缺值
    data['age'] = data['age'].replace(0, 'Unknown')
    # 对语言能力进行替换
    replace_dict = {3: 'high', 2: 'intermediate', 1: 'low', 0: 'no'}
    data.iloc[:, 6:12] = data.iloc[:, 6:12].map(replace_dict.get)
    # 准备记录结果
    id_pred_map = {id_: 0 for id_ in data['id'].unique()}
    token_count = []
    # 将ID信息转换为MD5码
    data['id'] = data['id'].apply(lambda x: int_md5_transform(num=x))
    education['id'] = education['id'].apply(lambda x: int_md5_transform(num=x))
    work['id'] = work['id'].apply(lambda x: int_md5_transform(num=x))
    # 遍历每个任务ID
    for i in range(len(data['task_id'].unique())):
        task_id = data['task_id'].unique()[i]
        numbers = []
        response = None
        count = data[data['task_id'] == task_id]['count'].max()
        for j in range(vote_count):
            retries = 0
            task_description = prompt_generator(data, education, work, task_id, prompt_type, stage=stage,
                                                shuffle=shuffle, model=model, client=client, detail=detail)
            while retries < 5:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f'{task_description}'},
                    ],
                    stream=False
                )
                match = re.search(r'@@@ Candidates selected: (.*?) @@@', response.choices[0].message.content)
                if match:
                    pred_numbers = re.findall(r'\b[a-f0-9]{6}\b', match.group(1))
                    if len(pred_numbers) == count:
                        numbers.append(pred_numbers)
                        break
                else:
                    retries += 1
        all_md5_hashes = [md5 for sublist in numbers for md5 in sublist]
        numbers = [md5 for md5, _ in Counter(all_md5_hashes).most_common(count)]
        token_count.append(response.usage.prompt_tokens)
        numbers = [int_md5_transform(md5_hash=hash_val, reverse=True) for hash_val in numbers]
        numbers.sort()
        for number in numbers:
            id_pred_map[number] = 1
        print(f'Current Progress: {round(((i+1) / len(data["task_id"].unique())) * 100, 1)}%\n '
              f'Task ID: {task_id} Candidates shortlisted are: {numbers}')
    # 提取结果
    pred = pd.DataFrame(list(id_pred_map.items()), columns=['id', 'pred'])
    data['id'] = data['id'].apply(lambda x: int_md5_transform(md5_hash=x, reverse=True))
    data = pd.merge(data, pred, on='id', how='left')
    # 评估结果
    acc = accuracy_score(data['interviewed'], data['pred'])
    f1 = f1_score(data['interviewed'], data['pred'])
    if save_result:
        # 将结果转为数据集
        log_result(data, stage, 'DeepSeek', f'{prompt_type}', token_count)

    return {'accuracy': acc, 'f1_score': f1}