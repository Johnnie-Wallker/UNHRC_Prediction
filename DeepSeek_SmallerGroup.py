import pandas as pd
import re
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from Prompt_Generator import prompt_generator
from Data_Handler import data_handler, int_md5_transform

# 读取数据
data = pd.read_excel('data.xlsx')
stage = 1
data = data_handler(data, stage)
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
id_pred_map = {id_: 0 for id_ in data['id'].unique()}
# 将ID信息转换为MD5码
data['id'] = data['id'].apply(lambda x: int_md5_transform(num=x))
education['id'] = education['id'].apply(lambda x: int_md5_transform(num=x))
work['id'] = work['id'].apply(lambda x: int_md5_transform(num=x))
# 配置API
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
prompt_type = 'None'
# 遍历每个任务ID
for i in range(len(data['task_id'].unique())):
    task_id = data['task_id'].unique()[i]
    response = None
    numbers = None
    task_df = data[data['task_id'] == task_id]
    count = data[data['task_id'] == task_id]['count'].max()
    group = task_df.sample(count + 1)
    task_df = task_df[~task_df['id'].isin(group['id'])]
    while len(task_df) != 0:
        task_description = prompt_generator(data, education, work, task_id, prompt_type, group)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": f'{task_description}'},
            ],
            stream=False
        )
        match = re.search(r'@@@ Candidates selected: (.*?) @@@', response.choices[0].message.content)
        numbers = re.findall(r'\b[a-f0-9]{6}\b', match.group(1))
        group = group[group['id'].isin(numbers)]
        new_data = task_df.sample(1)
        group = pd.concat([group, new_data])
        group = group.sample(frac=1).reset_index(drop=True)
        task_df = task_df[~task_df['id'].isin(new_data['id'])]
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
