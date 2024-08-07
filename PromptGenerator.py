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
    data.iloc[:, 5:25] = data.iloc[:, 5:25].astype('int')
    # 替换年龄空缺值
    data['age'] = data['age'].replace(0, 'Unknown')
    # 对语言能力进行替换
    replace_dict = {3: 'high', 2: 'intermediate', 1: 'low', 0: 'no'}
    data.iloc[:, 6:12] = data.iloc[:, 6:12].map(replace_dict.get)
    # 提取对应task_id的数据
    task_data = data[data['task_id'] == task_id]
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
        train_data = data[
            data['task_id'].isin(task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0])]
        for candidate_id in task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0]:
            task_data_new = data[data['task_id'] == candidate_id]
            edu_data_new = education[education['id'].isin(task_data_new['id'])]
            work_data_new = work[work['id'].isin(task_data_new['id'])]
            shortlisted_ids = task_data_new[task_data_new['interviewed'] == 1]['id'].tolist()
            information += (f"{candidate_information(task_data_new, edu_data_new, work_data_new)}"
                            f"The candidates that were shortlisted in this meeting are: "
                            f"{', '.join(map(str, shortlisted_ids))}.\n")
        if prompt_type == 'Train':
            row = task_data.iloc[0]
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'Your task is to rank these candidates based on how suitable you consider they are for this mandate.\n'
                f'Before making your rankings, in the previous years the council has held meetings on this mandate, '
                f'the candidates information in these meetings and the candidates that were shortlisted are:\n'
                f'{information}\n'
                f'Given the examples of previously successful candidates, '
                f'rank the candidates below from 1 to {len(task_data)}, where 1 stands for the most suitable '
                f'and {len(task_data)} stands for the least suitable.\n'
                f'Please respond with the following format: @@@ Rank: i Candidate ID: @@@, '
                f'do not include your reasons.\n'
                f'Before making your rankings, make sure you have carefully reviewed all the candidates below '
                f'regardless of the order they appear in the candidate information below.\n'
                f'The candidates information to be considered are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
            description += (f'IMPORTANT: Make sure you do not rank the candidates from the previous meetings '
                            f'when ranking the candidates, they are only there for referencing.')
        if prompt_type == 'Summary':
            row = task_data.iloc[0]
            client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                temperature=0.7,
                messages=[
                    {"role": "user", "content": f'You are an academic that is now studying what qualities a successful '
                                                f'candidate possess for {row["mandate"]} '
                                                f'in the United Nations Human Rights Council(UNHRC).'
                                                f'The candidates information in these UNHRC meetings are {information}.'
                                                f'Based on this information, summarise the keys features '
                                                f'of a successful candidate.\n'
                                                f'Focus (but not only focus) on the following points:\n'
                                                f'1. Is there a trend in age or language abilities '
                                                f'among the successful candidates?'
                                                f'2. Does certain gender has an advantage over the other?\n'
                                                f'3. Does certain nationality has an advantage over the others?\n'
                                                f'4. Does the legal tradition (Based on nationality, '
                                                f'location of education etc.) of the candidate make '
                                                f'the candidate more likely to be successful?\n'
                                                f'5. Does the location of education or the diversity of the location '
                                                f'of education(i.e. whether the university is in a OECD country or in '
                                                f'the global south) makes a candidate more likely to be successful?'},
                ],
                stream=False
            )
            summary = response.choices[0].message.content
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'Your task is to rank these candidates based on how suitable you consider they are for this mandate.\n'
                f'Before making your rankings, in the previous years the council has held meetings on this mandate, '
                f'here is a summary of the selected candidates in these previous UNHRC meetings:\n'
                f'{summary}\n'
                f'Referring to this information, '
                f'rank the candidates below from 1 to {len(task_data)}, where 1 stands for the most suitable '
                f'and {len(task_data)} stands for the least suitable.\n'
                f'Please respond with the following format: @@@ Rank: i Candidate ID: @@@, '
                f'do not include your reasons.\n'
                f'Before making your rankings, make sure you have carefully reviewed all the candidates below '
                f'regardless of the order they appear in the candidate information below.\n'
                f'The candidates information to be considered are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
        if prompt_type == 'RuleSet':
            ruleset = ruleset_generator(train_data)
            row = task_data.iloc[0]
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'Your task is to rank these candidates based on how suitable you consider they are for this mandate.\n'
                f'From previous knowledge, we know that {ruleset}\n'
                f'Referring to this information, '
                f'rank the candidates below from 1 to {len(task_data)}, where 1 stands for the most suitable '
                f'and {len(task_data)} stands for the least suitable.\n'
                f'Please respond with the following format: @@@ Rank: i Candidate ID: @@@, '
                f'do not include your reasons.\n'
                f'Before making your rankings, make sure you have carefully reviewed all the candidates below '
                f'regardless of the order they appear in the candidate information below.\n'
                f'The candidates information to be considered are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
        if prompt_type == 'Prototype':
            prototype = prototype_generator(train_data)
            row = task_data.iloc[0]
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'Your task is to rank these candidates based on how suitable you consider they are for this mandate.\n'
                f'From previous knowledge, we know that {prototype}\n'
                f'Referring to this information, '
                f'rank the candidates below from 1 to {len(task_data)}, where 1 stands for the most suitable '
                f'and {len(task_data)} stands for the least suitable.\n'
                f'Please respond with the following format: @@@ Rank: i Candidate ID: @@@, '
                f'do not include your reasons.\n'
                f'Before making your rankings, make sure you have carefully reviewed all the candidates below '
                f'regardless of the order they appear in the candidate information below.\n'
                f'The candidates information to be considered are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data) + "\n"
    else:
        row = task_data.iloc[0]
        description = (
            f'You are a member of the United Nations Human Rights Council(UNHRC), '
            f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
            f'Given the candidates information below, '
            f'please rank these candidates based on how suitable you consider they are for this mandate.\n'
            f'Rank them from 1 to {len(task_data)}, where 1 stands for the most suitable '
            f'and {len(task_data)} stands for the least suitable.\n'
            f'Please respond with the following format: @@@ Rank: i Candidate ID: id @@@, do not include your reasons.\n'
            f'Before making your rankings, make sure you have carefully reviewed all the candidates '
            f'regardless of the order they appear in the candidate information.\n'
            f'The candidates information to be considered are:\n'
        )
        description += candidate_information(task_data, edu_data, work_data) + "\n"

    return description