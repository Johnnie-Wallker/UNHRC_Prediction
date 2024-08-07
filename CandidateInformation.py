import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)


def candidate_information(task_data, edu_data, work_data):
    description = ''
    edu_data = edu_data[['id', 'degree_raw', 'major_final', 'university_final', 'country_final', 'years-final']]
    work_data = work_data[['id', 'title', 'orz', 'country_final', 'years']]
    edu_data = edu_data.dropna()
    work_data = work_data.dropna()
    # 将每名参选者的信息添加
    for _, row in task_data.iterrows():
        candidate_id = row['id']
        he_she = "He" if row['gender_final'] == "Male" else "She"
        his_her = "His" if row['gender_final'] == "Male" else "Her"
        nationality = row['nationality_final']
        if row['other nationality_final'] != 0:
            nationality += f' and {row["other nationality_final"]}'
        candidate_info = (
            f'Candidate ID: {row["id"]}, {he_she} is a citizen of {nationality}. '
            f'{his_her} age is {row["age"]}. '
        )

        # 语言能力
        languages = {
            'english_level': row['english_level'],
            'french_level': row['french_level'],
            'arabic_level': row['arabic_level'],
            'chinese_level': row['chinese_level'],
            'russian_level': row['russian_level'],
            'spanish_level': row['spanish_level']
        }

        language_info = ', '.join(
            [f"{level} {language.replace('_', ' ')}" for language, level in languages.items() if level != 'no'])
        candidate_info += f"{he_she} has {language_info}.\n"

        # # 背景信息
        # backgrounds = {
        #     'legal background': row['lawflag'],
        #     'religious background': row['church_flag'],
        #     'academic background': row['academic_flag'],
        #     'company background': row['firm_flag'],
        #     'IGO background': row['internationalorg_flag'],
        #     'NGO background': row['ngo_flag'],
        #     'government background': row['state_flag']
        # }
        #
        # background_info = ', '.join([f"{background}" for background, flag in backgrounds.items() if flag == 1])
        # if background_info:
        #     candidate_info += f"{he_she} has {background_info}. "
        #
        # # 当前类型
        # current_job_flags = {
        #     'academic': row['academic_current_flag'],
        #     'IGO': row['internationalorg_current_flag'],
        #     'NGO': row['ngo_current_flag'],
        #     'government': row['state_current_flag'],
        #     'company': row['firm_current_flag'],
        #     'religious': row['church_current_flag'],
        #     'law': row['la_current_flag']
        # }
        # current_jobs = [job for job, flag in current_job_flags.items() if flag == 1]
        #
        # if current_jobs:
        #     current_job_str = " and ".join(current_jobs)
        #     candidate_info += f'{his_her} current job is {current_job_str} related.\n'

        candidate_info += f'{his_her} educational background is as following:\n'

        for _, edu_row in edu_data.iterrows():
            if edu_row['id'] == candidate_id:
                candidate_info += (f'{edu_row["years-final"]}: w{edu_row["degree_raw"]} in {edu_row["major_final"]}, '
                                   f'{edu_row["university_final"]}, {edu_row["country_final"]}.\n')

        candidate_info += f'{his_her} working background is as following:\n'

        for _, work_row in work_data.iterrows():
            if work_row['id'] == candidate_id:
                candidate_info += (f'{work_row["years"]}: {work_row["title"]} of '
                                   f'{work_row["orz"]}, {work_row["country_final"]}.\n')

        description += candidate_info

    return description