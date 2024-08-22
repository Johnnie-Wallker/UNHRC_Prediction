import os
import pandas as pd
from CandidateInformation import candidate_information
from RuleSet import ruleset_generator
from Prototype import prototype_generator
pd.options.mode.copy_on_write = True


def relevant_information(data, education, work, task_id_map, detail, task_id):
    information = ''
    for candidate_id in task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0]:
        task_data_new = data[data['task_id'] == candidate_id]
        edu_data_new = education[education['id'].isin(task_data_new['id'])]
        work_data_new = work[work['id'].isin(task_data_new['id'])]
        shortlisted_ids = task_data_new[task_data_new['interviewed'] == 1]['id'].tolist()
        information += (f"{candidate_information(task_data_new, edu_data_new, work_data_new, detail)}"
                        f"The candidates that were shortlisted in this meeting are: "
                        f"{', '.join(map(str, shortlisted_ids))}.\n")
    return information


def summary_writer(information, mandate, task_id, detail, client, model, stage):
    folder_path = f'Summary/task_{task_id}/stage{stage}'
    os.makedirs(folder_path, exist_ok=True)
    if detail:
        file_path = os.path.join(folder_path, "summary_detail.txt")
    elif detail == 'Education':
        file_path = os.path.join(folder_path, "summary_edu.txt")
    elif detail == 'Work':
        file_path = os.path.join(folder_path, "summary_work.txt")
    else:
        file_path = os.path.join(folder_path, "summary.txt")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                summary = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as file:
                summary = file.read()
    else:
        prompt = (f'You are a researcher studying the qualities of successful candidates for {mandate} in the '
                  f'United Nations Human Rights Council. The candidates information in these UNHRC meetings are '
                  f'{information}\nReferring to this specific mandate, summarise the key features of the '
                  f'shortlisted candidates , ensure that your summary is not vague and ambiguous, this summary '
                  f'should aim to provide a clear guideline towards candidate selection process in the UNHRC. '
                  f'Do not mention the specific candidate IDs in your summary.')
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        summary = response.choices[0].message.content
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(summary)

    return summary


def prompt_generator(data, education, work, task_id, prompt_type, client, model, stage, detail, shuffle,
                     small_group=None):
    description = ''
    if small_group is not None:
        task_data = small_group.copy()
    else:
        task_data = data[data['task_id'] == task_id]
        if shuffle:
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
    row = task_data.iloc[0]
    if row['count'] != 1:
        guideline = f'@@@ Candidates selected: id1, ..., id{row["count"]} @@@,'
    else:
        guideline = f'@@@ Candidate selected: id @@@,'
    if prompt_type != 'None' and task_id in task_id_map.loc[:, 'task_id'].values:
        information = relevant_information(data, education, work, task_id_map, detail, task_id)
        train_data = data[
            data['task_id'].isin(task_id_map.loc[task_id_map['task_id'] == task_id, 'related_task_ids'].values[0])]
        if prompt_type == 'Train':
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'the information of the candidates in these meetings are:\n{information}'
                f'Now that you have had some experience, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate, '
                f'then select EXACTLY {row["count"]} candidate(s) that are to be shortlisted for interview.'
                f'Please respond with the following format: {guideline}'
                f'do not include your reasons for selecting them，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear and make sure you do not select'
                f'any candidate from any previous meetings, only consider those in the information below.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n!!!"
        if prompt_type == 'Summary':
            summary = summary_writer(information, row['mandate'], task_id, detail, client, model, stage)
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, here is a summary of '
                f'the key features of successful candidates in these meetings:\n{summary}\n'
                f'According to this summary, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate according to the mandate and summary, '
                f'then select EXACTLY {row["count"]} candidate(s) that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'Please respond with the following format: {guideline}'
                f'do not include your reasons for selecting them，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n"
        if prompt_type == 'Train+Summary':
            summary = summary_writer(information, row['mandate'], task_id, detail, client, model, stage)
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'the information of the candidates in these meetings are:\n{information}'
                f'In addition, scholars have summarised the key features of successful candidates in these meetings:\n'
                f'{summary}\nAccording to this information, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate according to the mandate and summary, '
                f'then select EXACTLY {row["count"]} candidate(s) that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'Please respond with the following format: {guideline}'
                f'do not include your reasons for selecting them，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear and make sure you do not select'
                f'any candidate from any previous meetings, only consider those in the information below.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n"
        if prompt_type == 'RuleSet':
            ruleset = ruleset_generator(train_data)
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'experts have found that: \n{ruleset}'
                f'Referring to this information, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate, '
                f'then select EXACTLY {row["count"]} candidate(s) that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'Please respond with the following format: {guideline}'
                f'do not include your reasons for selecting them，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n"
        if prompt_type == 'Prototype':
            prototype = prototype_generator(train_data)
            description = (
                f'You are a member of the United Nations Human Rights Council(UNHRC), '
                f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
                f'In the previous years, the council has held meetings on the same mandate, '
                f'experts have found that: \n{prototype}'
                f'Referring to this information, review the candidates curriculum information below, '
                f'summarise the strengths and characteristics of each candidate, '
                f'then select EXACTLY {row["count"]} candidate(s) that are to be shortlisted for interview.'
                f'Please respond with the following format: '
                f'Please respond with the following format: {guideline}'
                f'do not include your reasons for selecting them，\n'
                f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
                f'list of ALL the candidates below regardless of the order they appear.\n'
                f'The candidates information are:\n'
            )
            description += candidate_information(task_data, edu_data, work_data, detail) + "\n"
    else:
        description = (
            f'You are a member of the United Nations Human Rights Council(UNHRC), '
            f'the council is now holding a meeting for selecting {row["mandate"]}.\n'
            f'Review the candidates curriculum information below, summarise the strengths and characteristics of each '
            f'candidate, then select EXACTLY {row["count"]} candidate(s) that are to be shortlisted for interview.'
            f'Please respond with the following format: '
            f'Please respond with the following format: {guideline}'
            f'do not include your reasons for selecting them，\n'
            f'IMPORTANT: Before selecting any candidate, make sure you have thoroughly reviewed the entire '
            f'list of ALL the candidates below regardless of the order they appear.\n'
            f'The candidates information are:\n'
        )
        description += candidate_information(task_data, edu_data, work_data, detail) + "\n"

    return description
