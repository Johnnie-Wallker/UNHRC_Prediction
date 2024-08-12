import pandas as pd
import re
import warnings
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from Prompt_Generator import prompt_generator
from Data_Handler import data_handler, int_md5_transform
from Result_Logger import log_result
warnings.filterwarnings("ignore", category=UserWarning, message="Token indices sequence length is longer than the "
                                                                "specified maximum sequence length for this model")

# 读取数据
data = pd.read_excel('data.xlsx')
stage = 1
data = data_handler(data, stage)
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
id_pred_map = {id_: 0 for id_ in data['id'].unique()}
token_count = []
# 将ID信息转换为MD5码
data['id'] = data['id'].apply(lambda x: int_md5_transform(num=x))
education['id'] = education['id'].apply(lambda x: int_md5_transform(num=x))
work['id'] = work['id'].apply(lambda x: int_md5_transform(num=x))
# 配置API
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
prompt_type = 'Train'
# 遍历每个任务ID
for i in range(len(data['task_id'].unique())):
    task_id = data['task_id'].unique()[i]
    # 生成提示语并记录token数
    task_description = prompt_generator(data, education, work, task_id, prompt_type)
    # 大模型回复
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.7,
        messages=[
            {"role": "user", "content": f'{task_description}'},
        ],
        stream=False
    )
    token_count.append(response.usage.prompt_tokens)
    # 提取候选人ID
    numbers = re.findall(r'\b[a-f0-9]{6}\b',
                         re.search(r'@@@ Candidates selected: (.*?) @@@', response.choices[0].message.content).group(1))
    numbers = [int_md5_transform(md5_hash=hash_val, reverse=True) for hash_val in numbers]
    numbers.sort()
    for number in numbers:
        id_pred_map[number] = 1
    print(f'Current Progress: {round((i / len(data["task_id"].unique())) * 100, 1)}%\n '
          f'Task ID: {task_id} Candidates shortlisted are: {numbers}')

# 提取结果
pred = pd.DataFrame(list(id_pred_map.items()), columns=['id', 'pred'])
data['id'] = data['id'].apply(lambda x: int_md5_transform(md5_hash=x, reverse=True))
data = pd.merge(data, pred, on='id', how='left')
# 评估结果
acc = accuracy_score(data['interviewed'], data['pred'])
f1 = f1_score(data['interviewed'], data['pred'])
print(f'准确率为：{acc} 召回率为：{f1}')
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
