from kmodes.kprototypes import KPrototypes
import pandas as pd
from openai import OpenAI
pd.options.mode.copy_on_write = True


def prototype_generator(train_data):
    X = train_data.drop(['id', 'task_id', 'interviewed', 'mandate', 'nationality_final', 'count'], axis=1)
    X = X[[X.columns[1], X.columns[0]] + list(X.columns[2:])]
    y = train_data['interviewed']
    categorical_columns = [i for i in range(X.shape[1]) if i != 0]
    model = KPrototypes(2, random_state=1).fit(X, y, categorical=categorical_columns)
    centers = model.cluster_centroids_
    center = centers[1]
    non_zero_columns = {col: val for col, val in zip(X.columns, center) if val != 0}
    formatted_string = ', '.join([f'{col}:{val}' for col, val in non_zero_columns.items()])
    prototype = f'The candidate is more likely to be shortlisted if information is most similar to {formatted_string}'
    client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
    # 大模型回复
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=1,
        messages=[
            {"role": "user", "content": f'Please convert {prototype} into a concise paragraph, '
                                        f'generalise the condition rather than converting it straight away'
                                        f'(especially for continuous keys like age).\n'
                                        f'The meaning of the keys are:\n'
                                        f'Gender: 0 for female and 1 for male.\n'
                                        f'Language abilities: The values 0, 1, 2, 3 represent '
                                        f'No, Low, Intermediate and High respectively.\n'
                                        f'Flags: If there is no "current" in the key, then this represents whether '
                                        f'the candidate has this background(0 for no, 1 for yes). '
                                        f'If there is a "current" in the key, then this represents whether the current '
                                        f'job of this candidate is related to this field (0 for no, 1 for yes).\n'
                                        f'The flag name representations are: \n'
                                        f'law/la represents law; academic represents academic; '
                                        f'church represents religious, firm represents company, '
                                        f'state represents government, internationalorg represents IGO, '
                                        f'ngo represents NGO.\n'
                                        f'Please do not include the keys of the original condition '
                                        f'nor explain why this condition is true in any means, '
                                        f'simply rewrite the condition in plain '
                                        f'english text that is straightforward to anyone that does not know this data'
                                        f', replacing any keys with meaningful terms while making sure the keys you '
                                        f'interpret is the same as the key explanation that I have given you.\n'
                                        f'PLease respond with the following format: '
                                        f'Generally, a successful candidate is associated with:'},
        ],
        stream=False
    )
    llm_prototype = response.choices[0].message.content

    return llm_prototype