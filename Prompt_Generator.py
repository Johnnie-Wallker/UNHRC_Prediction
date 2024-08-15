import os
import pandas as pd
from openai import OpenAI
from CandidateInformation import candidate_information
from RuleSet import ruleset_generator
from Prototype import prototype_generator
pd.options.mode.copy_on_write = True


def summary_writer(information, mandate, task_id):
    folder_path = f'Summary/{task_id}'
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "summary_test2.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            summary = file.read()
    else:
        client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": f'You are a researcher trying to study the qualities of a successful '
                                            f'candidate for {mandate} in the United Nations Human Rights Council.'
                                            f'The candidates information in these UNHRC meetings are {information}.'
                                            f'Summarise the key features of successful candidates. \n'
                                            f'Make sure that no middle ground answers are acceptable, \n'
                                            f'since the entire council will rely on you to make their decisions. \n'
                                            f'You have to watch out for all kinds of, even the slightest differences among the candidates. \n'
                                            f'Focus but not only focus on the following points (be aware that since data is limited, every example provided above is valuable, and they reflect key features):\n'
                                            f'1. Is there a trend in age for the candidates? Is there an age range for the candidates being shortlisted? \n'
                                            f'2. Is there a trend in language abilities among the successful candidates? Are they typically required to be fluent in some particular major languages? \n'
                                            f'3. Does certain gender have an advantage over the other? Study the proportion of male and female shortlisted candidates based off the pronounce being used.\n'
                                            f'4. Does certain nationality or the regions in which the candidates were born \n'
                                            f'have an advantage over the others? Which regions produce the most successful candidates? \n'
                                            f'5. Does the legal tradition (Based on nationality, '
                                            f'location of education etc.) of the candidate make '
                                            f'the candidate more likely to be successful?\n'
                                            f'6. Does the location of education or the diversity of the location '
                                            f'of education(i.e. whether the university is in a OECD country or in '
                                            f'the global south) makes the candidate more likely to be successful?'},
            ],
            stream=False
        )
        summary = response.choices[0].message.content
        with open(file_path, 'w') as file:
            file.write(summary)

    return summary


def prompt_generator(data, education, work, task_id, prompt_type, small_group=None, detail=True):
    description = ''
    # 提取对应task_id的数据
    if small_group is not None:
        task_data = small_group.copy()
    else:
        task_data = data[data['task_id'] == task_id]
        # 打乱数据顺序
        task_data = task_data.sample(frac=1).reset_index(drop=True)
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
            information += (f"{candidate_information(task_data_new, edu_data_new, work_data_new, detail)}"
                            f"The candidates that were shortlisted in this meeting are: "
                            f"{', '.join(map(str, shortlisted_ids))}.\n")
        if prompt_type == 'Train':
            row = task_data.iloc[0]
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'the information of the candidates in these meetings are:\n{information}'
                f'Now that you have had some experience, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate, '
                f'then select EXACTLY {row["count"]} candidates that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'@@@ Candidates selected: id1, ..., id{row["count"]} @@@, '
                f'do not include your reasons for selecting them，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear and make sure you do not select'
                f'any candidate from any previous meetings, only consider those in the information below.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n!!!"
        if prompt_type == 'Summary':
            row = task_data.iloc[0]
            summary = summary_writer(information, row['mandate'], task_id)
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'scholars have summarised the key features of successful candidates in these meetings:\n{summary}'
                f'Referring to this summary, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate according to the mandate and summary, '
                f'then select EXACTLY {row["count"]} candidates that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'@@@ Candidates selected: id1, ..., id{row["count"]} @@@,'
                f'do not include your reasons for selecting them，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n"
        if prompt_type == 'RuleSet':
            ruleset = ruleset_generator(train_data)
            row = task_data.iloc[0]
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'experts have found that: \n{ruleset}'
                f'Referring to this information, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate, '
                f'then select EXACTLY {row["count"]} candidates that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'@@@ Candidates selected: id1, ..., id{row["count"]} @@@，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n"
        if prompt_type == 'Prototype':
            prototype = prototype_generator(train_data)
            row = task_data.iloc[0]
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'experts have found that: \n{prototype}'
                f'Referring to this information, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate, '
                f'then select EXACTLY {row["count"]} candidates that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'@@@ Candidates selected: id1, ..., id{row["count"]} @@@，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n"
    else:
        row = task_data.iloc[0]
        description = (
            f'You are a member of the United Nations Human Rights Council(UNHRC), '
            f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
            f'Review the candidates curriculum information below, summarise the strengths and characteristics of each '
            f'candidate, then select EXACTLY {row["count"]} candidates that are to be shortlisted for interview.'
            f'Please respond with the following format: '
            f'@@@ Candidates selected: id1, ..., id{row["count"]} @@@, '
            f'do not include your reasons for selecting them，\n'
            f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
            f'list of ALL the candidates below regardless of the order they appear.\n'
            f'The candidates information are:\n'
        )
        description += candidate_information(task_data, edu_data, work_data, detail) + "\n"

    return description