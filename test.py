from openai import OpenAI
from CandidateInformation import candidate_information
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.copy_on_write = True

# 读取数据
data = pd.read_csv('df.csv')
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
task_id = 86
description = ''
# 对语言能力进行替换
replace_dict = {3: 'high', 2: 'intermediate', 1: 'low', 0: 'no'}
data.iloc[:, 6:12] = data.iloc[:, 6:12].map(replace_dict.get)
# 提取对应task_id的数据
task_data = data[data['task_id'] == task_id]
# API构建
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
# 提取每个职位对应的全部task_id
mandate_groups = data.groupby('mandate')['task_id'].apply(lambda x: list(x.unique()))
# 过滤出多个task_id的职位
mandate_groups = mandate_groups[mandate_groups.apply(len) > 1]
# 获得与每个task_id相同职位的task_id
task_id_map = {task_id: [tid for tid in task_ids if tid != task_id]
               for task_ids in mandate_groups for task_id in task_ids}
task_id_map = pd.DataFrame(task_id_map.items(), columns=['task_id', 'related_task_ids'])
information = ''
for candidate_id in task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0]:
    task_data_new = data[data['task_id'] == candidate_id]
    edu_data_new = education[education['id'].isin(task_data_new['id'])]
    work_data_new = work[work['id'].isin(task_data_new['id'])]
    shortlisted_ids = task_data_new[task_data_new['interviewed'] == 1]['id'].tolist()
    information += (f"{candidate_information(task_data_new, edu_data_new, work_data_new)}"
                    f"In this case, the candidates that were shortlisted are: "
                    f"{', '.join(map(str, shortlisted_ids))}.\n")
row = task_data.iloc[0]
# 大模型回复
response = client.chat.completions.create(
    model="deepseek-chat",
    temperature=1,
    messages=[
        {"role": "user", "content": f'Using the candidates information below, '
                                    f'please summarise what makes a candidate suitable for {row["mandate"]} '
                                    f'given {information}.'
                                    f'Do not include the candidates ID nor their exact information in the summary, '
                                    f'summarise key attributes that make these candidates suitable for this mandate.'},
    ],
    stream=False
)
summary = response.choices[0].message.content
print(summary)
