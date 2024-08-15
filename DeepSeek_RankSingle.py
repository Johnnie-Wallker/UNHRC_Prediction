import pandas as pd
import re
from openai import OpenAI
from Data_Handler import data_handler
from CandidateInformation import single_candidate_information
pd.options.mode.copy_on_write = True

# 读取数据
data = pd.read_excel('data.xlsx')
stage = 1
data = data_handler(data, stage)
education = pd.read_excel('data.xlsx', sheet_name=1)
work = pd.read_excel('data.xlsx', sheet_name=2)
client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
id_pred_map = {id_: 0 for id_ in data['id'].unique()}
for i in range(len(data)):
    candidate_id = data['id'].unique()[i]
    candidate_information = single_candidate_information(data, education, work, candidate_id)
    prompt = (f'You are a member of the United Nation Human Rights Council, consider the following candidate '
              f'for mandate: {data.loc[data["id"] == candidate_id, "mandate"].values[0]}.\n{candidate_information}'
              f'After reviewing the curriculum information of this candidate, please score this candidate from 1 to 100'
              f' based on how well you consider this candidate fits with the given mandate, '
              f'a higher score means the candidate is more suitable for this mandate. Make sure you have fully '
              f'considered all aspects of this candidate, there will be multiple candidates applying for this mandate '
              f'and the competition is fierce since all candidates will be experts in related fields. '
              f'Your score will be used to compare these candidates so make sure the score you give is '
              f'critical and strict, you should divide the candidate information into several aspects and '
              f'give a smaller number on each aspect and take the sum as the final score you give(but make sure the '
              f'maximum score adds up to 100 and each aspect is weighted according to its importance). '
              f'For the rating you give, please use the following format: @@@ Rating: @@@')
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    answer = int(re.findall(r'@@@.*?(\d+).*?@@@', response.choices[0].message.content)[0])
    print(f'The rating for candidate {candidate_id} is {answer}.')
    id_pred_map[candidate_id] = answer