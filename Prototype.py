from kmodes.kprototypes import KPrototypes
import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.optimize import linear_sum_assignment
pd.options.mode.copy_on_write = True


def compute_distance(instance, centroid, categorical_indices, gamma):
    num_distance = np.sum((instance[~categorical_indices] - centroid[~categorical_indices]) ** 2)
    cat_distance = np.sum(instance[categorical_indices] != centroid[categorical_indices])

    return num_distance + gamma * cat_distance


def class_locater(model, x, y, categorical_columns):
    gamma = model.gamma
    centers = model.cluster_centroids_
    categorical_indices = np.zeros(x.shape[1], dtype=bool)
    categorical_indices[categorical_columns] = True
    class_distances = {i: [] for i in range(model.n_clusters)}
    for cls in np.unique(y):
        class_instances = x.iloc[np.where(y == cls)[0], :]
        class_instances = class_instances.to_numpy()
        distances = np.zeros((len(class_instances), model.n_clusters))
        for i, instance in enumerate(class_instances):
            for j, centroid in enumerate(centers):
                distances[i, j] = compute_distance(instance, centroid, categorical_indices, gamma)
        for i in range(model.n_clusters):
            class_distances[i].append((cls, np.mean(distances[:, i])))
    cost_matrix = np.zeros((model.n_clusters, len(np.unique(y))))
    classes = list(np.unique(y))
    for i in range(model.n_clusters):
        for j, cls in enumerate(classes):
            for class_distance in class_distances[i]:
                if class_distance[0] == cls:
                    cost_matrix[i, j] = class_distance[1]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    closest_classes = {row: classes[col] for row, col in zip(row_ind, col_ind)}

    return closest_classes


def prototype_generator(train_data):
    X = train_data.drop(['id', 'task_id', 'interviewed', 'mandate', 'nationality_final', 'count'], axis=1)
    X = X[[X.columns[1], X.columns[0]] + list(X.columns[2:])]
    y = train_data['interviewed']
    categorical_columns = [i for i in range(X.shape[1]) if i != 0]
    model = KPrototypes(2, random_state=1).fit(X, y, categorical=categorical_columns)
    centers = model.cluster_centroids_
    classes = class_locater(model, X, y, categorical_columns)
    X_columns = X.columns
    columns1 = {col: val for col, val in zip(X_columns, centers[classes[1]])}
    formatted_string1 = ', '.join([f'{col}:{val}' for col, val in columns1.items()])
    columns2 = {col: val for col, val in zip(X_columns, centers[classes[0]])}
    formatted_string2 = ', '.join([f'{col}:{val}' for col, val in columns2.items()])
    prototype = (f'The candidate is most likely to be shortlisted if '
                 f'information is most similar to {formatted_string1}\n')
    prototype += (f'The candidate is most likely to not be shortlisted if '
                  f'information is most similar to {formatted_string2}')
    client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
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
