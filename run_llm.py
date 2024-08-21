from DeepSeek import run_deepseek
from openai import OpenAI

config = {
    'prompt_type': 'None',
    'stage':1,
    'shuffle': False,
    'detail': False,
    'vote_count': 1,
    'n_retry': 5,
    'do_small_group': True,
    'client': OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com"),
    'model': "deepseek-chat"
}

results = run_deepseek(**config, save_result=False)
print(f'准确率为：{round(results["accuracy"], 3)} 召回率为：{round(results["f1_score"], 3)}')