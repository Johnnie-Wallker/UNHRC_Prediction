from LLM import run_llm
from ML import run_ml
from openai import OpenAI
from xgboost import XGBClassifier

config = {
    'stage': 1,
    'detail': True,
    'model': "deepseek-chat",  # XGBClassifier(objective='binary:logistic', seed=1, enable_categorical=True)
    'save_result': False,
    'prompt_type': 'Train',
    'shuffle': False,
    'vote_count': 1,
    'n_retry': 10,
    'do_small_group': False,
    'smallgroup_threshold': 30,
    'client': OpenAI(api_key="sk-668cbc8b98014bc29f460fe20ff7a225", base_url="https://api.deepseek.com"),
}

if isinstance(config['model'], str):
    results = run_llm(**config)
else:
    results = run_ml(**config)
print(f'准确率为：{round(results["accuracy"], 3)} 召回率为：{round(results["f1_score"], 3)}')