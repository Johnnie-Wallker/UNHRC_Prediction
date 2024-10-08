import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score
from Prompt_Generator import prompt_generator
from Data_Handler import data_handler, int_md5_transform
from Result_Logger import log_result
from collections import Counter


def run_llm(config):
    data = pd.read_excel('data.xlsx')
    data = data_handler(data, config['stage'])
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    education = pd.read_excel('data.xlsx', sheet_name=1)
    work = pd.read_excel('data.xlsx', sheet_name=2)
    data = data.fillna(0)
    data.iloc[:, 5:25] = data.iloc[:, 5:25].astype('int')
    data['age'] = data['age'].replace(0, 'Unknown')
    replace_dict = {3: 'high', 2: 'intermediate', 1: 'low', 0: 'no'}
    data.iloc[:, 6:12] = data.iloc[:, 6:12].map(replace_dict.get)
    id_pred_map = {id_: 0 for id_ in data['id'].unique()}
    token_count = []
    data['id'] = data['id'].apply(lambda x: int_md5_transform(num=x))
    education['id'] = education['id'].apply(lambda x: int_md5_transform(num=x))
    work['id'] = work['id'].apply(lambda x: int_md5_transform(num=x))
    for i in range(len(data['task_id'].unique())):
        task_id = data['task_id'].unique()[i]
        count = data[data['task_id'] == task_id]['count'].max()
        response = None
        numbers = []
        for j in range(config['vote_count']):
            task_df = data[data['task_id'] == task_id]
            retries = 0
            if not config['do_small_group'] or len(task_df) <= min(config['smallgroup_threshold'], count * 2):
                task_description = prompt_generator(data=data,
                                                    education=education,
                                                    work=work,
                                                    task_id=task_id,
                                                    prompt_type=config['prompt_type'],
                                                    stage=config['stage'],
                                                    shuffle=config['shuffle'],
                                                    model=config['model'],
                                                    client=config['client'],
                                                    detail=config['detail'])
                while retries < config['n_retry']:
                    response = config['client'].chat.completions.create(
                        model=config['model'],
                        messages=[
                            {"role": "user", "content": f'{task_description}'},
                        ],
                        stream=False
                    )
                    match = re.search(r'selected: (.*?)( @@@|$)', response.choices[0].message.content)
                    if match is not None:
                        pred_numbers = re.findall(r'\b[a-f0-9]{6}\b', match.group(1))
                        if len(pred_numbers) == count:
                            numbers.append(pred_numbers)
                            break
                        else:
                            retries += 1
                            print(f'ERROR: The response given is: {response.choices[0].message.content}\n'
                                  f'The number of candidates given is incorrect({count} candidates needed), '
                                  f'retrying......')
                    else:
                        retries += 1
                        print(f'ERROR: The response given is: {response.choices[0].message.content}\n'
                              f'Response format is wrong, retrying......')
            else:
                group = task_df.sample(count*2)
                task_df = task_df[~task_df['id'].isin(group['id'])]
                numbers_temp = None
                while len(task_df) != 0:
                    task_description = prompt_generator(data=data,
                                                        education=education,
                                                        work=work,
                                                        task_id=task_id,
                                                        prompt_type=config['prompt_type'],
                                                        stage=config['stage'],
                                                        shuffle=config['shuffle'],
                                                        model=config['model'],
                                                        client=config['client'],
                                                        detail=config['detail'],
                                                        small_group=group)
                    while retries < config['n_retry']:
                        response = config['client'].chat.completions.create(
                            model=config['model'],
                            messages=[
                                {"role": "user", "content": f'{task_description}'},
                            ],
                            stream=False
                        )
                        match = re.search(r'selected: (.*?)( @@@|$)', response.choices[0].message.content)
                        if match:
                            numbers_temp = re.findall(r'\b[a-f0-9]{6}\b', match.group(1))
                            if len(numbers_temp) == count:
                                numbers.append(numbers_temp)
                                break
                            else:
                                retries += 1
                                print(f'ERROR: The response given is: {response.choices[0].message.content}\n'
                                      f'The number of candidates given is incorrect({count} candidates needed), '
                                      f'retrying......')
                        else:
                            retries += 1
                            print(f'ERROR: The response given is: {response.choices[0].message.content}\n'
                                  f'Response format is wrong, retrying......')
                    group = group[group['id'].isin(numbers_temp)]
                    if len(task_df) >= count:
                        new_data = task_df.sample(count)
                    else:
                        new_data = task_df
                    group = pd.concat([group, new_data])
                    group = group.sample(frac=1).reset_index(drop=True)
                    task_df = task_df[~task_df['id'].isin(new_data['id'])]
                numbers.append(numbers_temp)
        all_md5_hashes = [md5 for sublist in numbers for md5 in sublist]
        numbers = [md5 for md5, _ in Counter(all_md5_hashes).most_common(count)]
        token_count.append(response.usage.prompt_tokens)
        numbers = [int_md5_transform(md5_hash=hash_val, reverse=True) for hash_val in numbers]
        numbers.sort()
        for number in numbers:
            id_pred_map[number] = 1
        print(f'Current Progress: {round(((i + 1) / len(data["task_id"].unique())) * 100, 1)}%\n '
              f'Task ID: {task_id} Candidates shortlisted are: {numbers}')
    pred = pd.DataFrame(list(id_pred_map.items()), columns=['id', 'pred'])
    data['id'] = data['id'].apply(lambda x: int_md5_transform(md5_hash=x, reverse=True))
    data = pd.merge(data, pred, on='id', how='left')
    acc = accuracy_score(data['interviewed'], data['pred'])
    f1 = f1_score(data['interviewed'], data['pred'])
    if config['save_result']:
        suffix = config['prompt_type']
        if config['do_small_group']:
            suffix += f'_SmallGroup'
        log_result(data, config['stage'], 'DeepSeek', suffix, token_count)

    return {'accuracy': acc, 'f1_score': f1}