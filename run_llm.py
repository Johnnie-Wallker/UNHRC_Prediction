from DeepSeek import run_deepseek
from openai import OpenAI

config = {
    'prompt_type': 'None',
    'stage': 1,
    'shuffle': False,
    'detail': 'Work',
    'vote_count': 1,
    'n_retry': 10,
    'do_small_group': False,
    'client': OpenAI(api_key="sk-668cbc8b98014bc29f460fe20ff7a225", base_url="https://api.deepseek.com"),
    'model': "deepseek-chat"
}

results = run_deepseek(**config, save_result=False)
print(f'准确率为：{round(results["accuracy"], 3)} 召回率为：{round(results["f1_score"], 3)}')