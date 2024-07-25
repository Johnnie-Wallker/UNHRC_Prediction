import warnings
import pandas as pd
from openai import OpenAI
from CandidateInformation import candidate_information
from RuleSet import ruleset_generator
from Prototype import prototype_generator
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.copy_on_write = True


def prompt_generator(data, education, work, task_id, prompt_type):
    description = ''
    # 填充空缺值
    data = data.fillna(0)
    data.iloc[:, 4:24] = data.iloc[:, 4:24].astype('int')
    # 提取对应task_id的数据
    task_data = data[data['task_id'] == task_id]
    # 替换年龄空缺值
    task_data['age'] = task_data['age'].replace(0, 'Unknown')
    # 对语言能力进行替换
    replace_dict = {3: 'high', 2: 'intermediate', 1: 'low', 0: 'no'}
    task_data.iloc[:, 5:11] = task_data.iloc[:, 5:11].map(replace_dict.get)
    edu_data = education[education['id'].isin(task_data['id'])]
    work_data = work[work['id'].isin(task_data['id'])]
    # 提取每个职位对应的全部task_id
    mandate_groups = data.groupby('mandate')['task_id'].apply(lambda x: list(x.unique()))
    # 过滤出多个task_id的职位
    mandate_groups = mandate_groups[mandate_groups.apply(len) > 1]
    # 获得与每个task_id相同职位的task_id
    task_id_map = {task_id: [tid for tid in task_ids if tid != task_id]
                   for task_ids in mandate_groups for task_id in task_ids}
    task_id_map = pd.DataFrame(task_id_map.items(), columns=['task_id', 'related_task_ids'])
    if prompt_type != 'None' and task_id in task_id_map.loc[:, 'task_id'].values:
        information = ''
        for candidate_id in task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0]:
            task_data_new = data[data['task_id'] == candidate_id]
            edu_data_new = education[education['id'].isin(task_data_new['id'])]
            work_data_new = work[work['id'].isin(task_data_new['id'])]
            shortlisted_ids = task_data_new[task_data_new['interviewed'] == 1]['id'].tolist()
            information += (f"{candidate_information(task_data_new, edu_data_new, work_data_new)}"
                            f"In this case, the candidates that were shortlisted are: "
                            f"{', '.join(map(str, shortlisted_ids))}.\n")
        if prompt_type == 'Train':
            # 生成提示语开头
            row = task_data.iloc[0]
            description = (
                f'You are a member of the UNHRC, based on the information of candidates, '
                f'select who can be shortlisted for interview. Their mandate is {row["mandate"]}.\n'
                f'Before selecting the candidates, this mandate has been mentioned in previous UNHRC meetings.\n'
                f'The candidates information in these meetings are:\n'
                f'{information}\n'
                f'Now that you have had some experience with this task, '
                f'please select "EXACTLY {row["count"]} candidates" using the candidates information below.\n'
                f'Give the ID numbers of the candidates that you have selected, '
                f'do not explain why you have chosen the candidates nor rank them in order, just the ID numbers.\n'
                f'Please respond with the following format: @@@ The candidates ID that you have selected are: @@@\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
            description += (f'Remember you need to select EXACTLY {row["count"]} and only {row["count"]} '
                            f'candidates from the candidates information above.')
        if prompt_type == 'Summary':
            row = task_data.iloc[0]
            client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
            # 大模型回复
            response = client.chat.completions.create(
                model="deepseek-chat",
                temperature=0.7,
                messages=[
                    {"role": "user", "content": f'Using the candidates information below, '
                                                f'please summarise what makes a candidate '
                                                f'suitable for {row["mandate"]} '
                                                f'given {information}.'
                                                f'Do not include the candidates ID nor their exact '
                                                f'information in the summary, '
                                                f'summarise key attributes that make these '
                                                f'candidates suitable for this mandate.'},
                ],
                stream=False
            )
            summary = response.choices[0].message.content
            # 生成提示语开头
            description = (
                f'You are a member of the UNHRC, based on the information of candidates, '
                f'select who can be shortlisted for interview. Their mandate is {row["mandate"]}.\n'
                f'Before selecting the candidates, this mandate has been mentioned in previous UNHRC meetings.\n'
                f'Here is the commentary summary of the selected candidates in these previous UNHRC meetings:\n'
                f'{summary}\n'
                f'Referring to this information, please select "EXACTLY {row["count"]} candidates" '
                f'using the candidates information below.\n'
                f'Give the ID numbers of the candidates that you have selected, '
                f'do not explain why you have chosen the candidates nor rank them in order, just the ID numbers.\n'
                f'Please respond with the following format: @@@ The candidates ID that you have selected are: @@@\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
            description += (f'Remember you need to select EXACTLY {row["count"]} and only {row["count"]} '
                            f'candidates from the candidates information above.')
        if prompt_type == 'RuleSet':
            train_data = data[
                data['task_id'].isin(task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0])]
            ruleset = ruleset_generator(train_data)
            # 生成提示语开头
            row = task_data.iloc[0]
            description = (
                f'You are a member of the UNHRC, based on the information of candidates, '
                f'select who can be shortlisted for interview. Their mandate is {row["mandate"]}.\n'
                f'From previous knowledge, we know that {ruleset} '
                f'Take this as a reference (and only as a reference) for selecting the suitable candidates.\n'
                f'Please select "EXACTLY {row["count"]} candidates" using the candidates information below.\n'
                f'Give the ID numbers of the candidates that you have selected, '
                f'do not explain why you have chosen the candidates nor rank them in order, just the ID numbers.\n'
                f'Please respond with the following format: @@@ The candidates ID that you have selected are: @@@\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
            description += (f'Remember you need to select EXACTLY {row["count"]} and only {row["count"]} '
                            f'candidates from the candidates information above.')
        if prompt_type == 'Prototype':
            train_data = data[
                data['task_id'].isin(task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0])]
            prototype = prototype_generator(train_data)
            # 生成提示语开头
            row = task_data.iloc[0]
            description = (
                f'You are a member of the UNHRC, based on the information of candidates, '
                f'select who can be shortlisted for interview. Their mandate is {row["mandate"]}.\n'
                f'From previous knowledge, we know that {prototype} '
                f'Take this as a reference (and only as a reference) for selecting the suitable candidates.\n'
                f'Please select "EXACTLY {row["count"]} candidates" using the candidates information below.\n'
                f'Give the ID numbers of the candidates that you have selected, '
                f'do not explain why you have chosen the candidates nor rank them in order, just the ID numbers.\n'
                f'Please respond with the following format: @@@ The candidates ID that you have selected are: @@@\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
            description += (f'Remember you need to select EXACTLY {row["count"]} and only {row["count"]} '
                            f'candidates from the candidates information above.')
    else:
        # 生成提示语开头
        row = task_data.iloc[0]
        description = (
            f'You are a member of the UNHRC, based on the information of candidates, '
            f'select who can be shortlisted for interview. Their mandate is {row["mandate"]}.\n'
            f'Please select "EXACTLY {row["count"]} candidates" using the candidates information below.\n'
            f'Give the ID numbers of the candidates that you have selected, '
            f'do not explain why you have chosen the candidates nor rank them in order, just the ID numbers.\n'
            f'Please respond with the following format: @@@ The candidates ID that you have selected are: @@@\n'
            f'The candidates information are:\n'
        )
        description += candidate_information(task_data, edu_data, work_data) + "\n"
        description += (f'Remember you need to select EXACTLY {row["count"]} and only {row["count"]} '
                        f'candidates from the candidates information above.')

    return description