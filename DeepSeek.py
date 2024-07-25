from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from PromptGenerator import prompt_generator
from DataHandler import data_handler
from ID_Finder import id_finder
import pandas as pd
import re

# 读取数据
data = pd.read_excel('data.xlsx')
stage = 1
data = data_handler(data, stage)
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
# API构建
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
# 创建空白字典
id_pred_map = {id_: 0 for id_ in data['id'].unique()}

# 遍历每个任务ID
for task_id in data['task_id'].unique():
    # 生成提示语
    task_description = prompt_generator(data, education, work, task_id, prompt_type='RuleSet')
    # 大模型回复
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.7,
        messages=[
            {"role": "user", "content": f'{task_description}'},
        ],
        stream=False
    )
    # 提取候选人ID
    numbers = re.findall(r'\d+', response.choices[0].message.content)
    for number in numbers:
        id_pred_map[int(number)] = 1
    print(task_id)

# 创建结果DataFrame
pred = pd.DataFrame(list(id_pred_map.items()), columns=['id', 'pred'])
# 合并原始数据和最终结果
data = pd.merge(data, pred, on='id', how='left')
data['pred'] = data['pred'].astype(int)

# 计算准确率
acc = accuracy_score(data['interviewed'], data['pred'])
f1 = f1_score(data['interviewed'], data['pred'])
print(f'准确率为：{acc} 召回率为：{f1}')
# 将结果转为数据集
id_finder(data, 'DeepSeek_RuleSet', stage)

# 简历筛选轮：
# 1.无样例 准确率为：0.6465661641541038 召回率为：0.4664981036662453
# 2.有样例 准确率为：0.6440536013400335 召回率为：0.46270543615676357
# 3.无样例(细节) 准确率为：0.7018425460636516 召回率为：0.549936788874842
# 4.有样例(细节) 准确率为：0.7043551088777219 召回率为：0.5537294563843237
# 5.有总结文本(细节) 准确率为：0.6859296482412061 召回率为：0.5259165613147914
# 6.RuleSet(细节) 准确率为：0.6595477386934674 召回率为：0.4857685009487666

# 面试轮：
# 1.无样例 准确率为：0.6506024096385542 召回率为：0.22459893048128343
# 2.有样例 准确率为：0.6506024096385542 召回率为：0.22459893048128343
# 3.无样例(细节) 准确率为：0.6602409638554216 召回率为：0.24598930481283424
# 4.有样例(细节) 准确率为：0.6867469879518072 召回率为：0.3048128342245989
# 5.有总结文本(细节) 准确率为：0.6987951807228916 召回率为：0.3315508021390374
# 6.RuleSet(细节) 准确率为：0.6698795180722892 召回率为：0.26737967914438504
