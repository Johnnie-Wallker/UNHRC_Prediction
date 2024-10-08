import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from openai import OpenAI


def tree_to_conditions(tree, feature_names, target_class=1):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    conditions = []

    def recurse(node, depth, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            path_left = path.copy()
            path_right = path.copy()
            path_left.append(f"(X['{name}'] <= {threshold})")
            path_right.append(f"(X['{name}'] > {threshold})")
            recurse(tree_.children_left[node], depth + 1, path_left)
            recurse(tree_.children_right[node], depth + 1, path_right)
        else:
            if np.argmax(tree_.value[node]) == target_class:
                condition = " & ".join(path)
                conditions.append(condition)
    recurse(0, 1, [])
    return conditions


def ruleset_generator(train_data):
    X = train_data.drop(['id', 'mandate', 'nationality_final', 'task_id', 'interviewed', 'count'], axis=1)
    y = train_data['interviewed']
    tree = DecisionTreeClassifier(max_depth=3, random_state=1)
    tree.fit(X, y)
    conditions = tree_to_conditions(tree, X.columns, target_class=1)
    condition = conditions[0]
    df = train_data[eval(condition)]
    prob = round(df['interviewed'].sum()/len(df), 2)
    rule = f"If {condition}, the probability of being shortlisted is higher({prob})."
    client = OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com")
    # 大模型回复
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=1,
        messages=[
            {"role": "user", "content": f'Please convert {rule} into a concise paragraph.\n'
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
                                        f', replacing any keys with meaningful terms.'},
        ],
        stream=False
    )
    llm_rule = response.choices[0].message.content

    return llm_rule
