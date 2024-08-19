import pandas as pd
import re
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from Prompt_Generator import prompt_generator
from Data_Handler import data_handler, int_md5_transform
from Result_Logger import log_result

# 设置参数
prompt_type = 'None'
stage = 1
shuffle = False
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
model = "deepseek-chat"
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
    response = None
    numbers = None
    task_df = data[data['task_id'] == task_id]
    count = data[data['task_id'] == task_id]['count'].max()
    group = task_df.sample(count + 1)
    task_df = task_df[~task_df['id'].isin(group['id'])]
    retries = 0
    if len(task_df) == 0:
        retries = 0
        task_description = prompt_generator(data, education, work, task_id, prompt_type, stage=stage,
                                            shuffle=shuffle, small_group=group, model=model, client=client)
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
                numbers = re.findall(r'\b[a-f0-9]{6}\b', match.group(1))
                break
            else:
                retries += 1
    else:
        while len(task_df) != 0:
            retries = 0
            task_description = prompt_generator(data, education, work, task_id, prompt_type, stage=stage,
                                                shuffle=shuffle, small_group=group, model=model, client=client)
            while retries < 5:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": f'{task_description}'},
                    ],
                    stream=False
                )
                match = re.search(r'@@@ Candidates selected: (.*?) @@@', response.choices[0].message.content)
                if match:
                    numbers = re.findall(r'\b[a-f0-9]{6}\b', match.group(1))
                    break
                else:
                    retries += 1
            group = group[group['id'].isin(numbers)]
            new_data = task_df.sample(1)
            group = pd.concat([group, new_data])
            group = group.sample(frac=1).reset_index(drop=True)
            task_df = task_df[~task_df['id'].isin(new_data['id'])]
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
print(f'准确率为：{round(acc, 3)} 召回率为：{round(f1, 3)}')
# 将结果转为数据集
log_result(data, stage, 'DeepSeek', f'{prompt_type}_SmallGroup', token_count)

# 准确率为：0.678 召回率为：0.515