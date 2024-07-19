from openai import OpenAI
from sklearn.metrics import accuracy_score
from PromptGenerator import prompt_generator
from ID_Finder import id_finder
import pandas as pd
import re

# 读取数据
data = pd.read_csv('df.csv')
data = data[data['interviewed'] == 1]
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
# API构建
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
# 创建空白字典
id_pred_map = {id_: 0 for id_ in data['id'].unique()}

# 遍历每个任务ID
for task_id in data['task_id'].unique():
    # 生成提示语
    task_description = prompt_generator(data, education, work, task_id, stage=1, train=True, summary=True)
    # 大模型回复
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.2,
        messages=[
            {"role": "user", "content": f'{task_description}'},
        ],
        stream=False
    )
    # 提取候选人ID
    numbers = re.findall(r'\d+', response.choices[0].message.content)
    for number in numbers:
        id_pred_map[int(number)] = 1

# 创建结果DataFrame
pred = pd.DataFrame(list(id_pred_map.items()), columns=['id', 'pred'])
# 合并原始数据和最终结果
data = pd.merge(data, pred, on='id', how='left')
data['pred'] = data['pred'].astype(int)

# 计算准确率
acc = accuracy_score(data['selected'], data['pred'])
print('准确率为：', acc)
# # 将结果转为数据集
# result_ds = id_finder(data, 1, 'DeepSeek')
# result_ds.to_csv('result_DeepSeek_Train.csv', index=False)
# 1轮无样例准确率: 0.6536565324568612
# 1轮无样例准确率(细节)： 0.6774856203779787
# 1轮有样例准确率: 0.6450287592440427
# 1轮有样例准确率(细节)： 0.6713229252259655
# 2轮无样例准确率: 0.6798088410991637
# 2轮无样例准确率(细节): 0.6833930704898447
# 2轮有样例准确率: 0.6738351254480287
# 2轮有样例准确率(细节): 准确率为： 0.6690561529271206
