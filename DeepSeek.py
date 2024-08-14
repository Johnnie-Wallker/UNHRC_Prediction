import pandas as pd
import re
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from Prompt_Generator import prompt_generator
from Data_Handler import data_handler, int_md5_transform
from Result_Logger import log_result
from collections import Counter

# 读取数据
data = pd.read_excel('data.xlsx')
stage = 1
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
# 配置API
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
prompt_type = 'Summary'
# 遍历每个任务ID
for i in range(len(data['task_id'].unique())):
    task_id = data['task_id'].unique()[i]
    numbers = []
    response = None
    count = data[data['task_id'] == task_id]['count'].max()
    for j in range(1):
        retries = 0
        task_description = prompt_generator(data, education, work, task_id, prompt_type)
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
                numbers.append(re.findall(r'\b[a-f0-9]{6}\b', match.group(1)))
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
print(f'准确率为：{round(acc, 3)} 召回率为：{round(f1, 3)}')
# 将结果转为数据集
log_result(data, stage, 'DeepSeek', prompt_type, token_count)

# 简历筛选轮：
# 1.无样例 准确率为：0.6499162479061976 召回率为：0.4715549936788875
# 2.有样例 准确率为：0.6390284757118928 召回率为：0.45512010113780027
# 3.无样例(细节) 准确率为：0.7018425460636516 召回率为：0.549936788874842
# 4.有样例(细节) 准确率为：0.7043551088777219 召回率为：0.5537294563843237
# 5.有总结文本(细节) 准确率为：0.6934673366834171 召回率为：0.5372945638432364
# 6.RuleSet(细节) 准确率为：0.6595477386934674 召回率为：0.4857685009487666
# 7.Prototype(细节) 准确率为：0.6654103852596315 召回率为：0.4946236559139785

# 面试轮：
# 1.无样例 准确率为：0.6506024096385542 召回率为：0.22459893048128343
# 2.有样例 准确率为：0.6506024096385542 召回率为：0.22459893048128343
# 3.无样例(细节) 准确率为：0.6602409638554216 召回率为：0.24598930481283424
# 4.有样例(细节) 准确率为：0.6867469879518072 召回率为：0.3048128342245989
# 5.有总结文本(细节) 准确率为：0.6987951807228916 召回率为：0.3315508021390374
# 6.RuleSet(细节) 准确率为：0.6698795180722892 召回率为：0.26737967914438504

# 简历筛选轮：
# 无样例（10次投票） 准确率为：0.685 召回率为：0.524
# 有样例（10次投票） 准确率为：0.683 召回率为：0.519
# 无样例（20次投票） 准确率为：0.704 召回率为：0.552