from DeepSeek import run_deepseek
from openai import OpenAI

config = {
    'prompt_type': 'Summary',
    'stage':1,
    'shuffle': False,
    'detail': True,
    'vote_count': 1,
    'client': OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com"),
    'model': "deepseek-chat"
}

results = run_deepseek(config, save_result=True)
print(f'准确率为：{round(results["accuracy"], 3)} 召回率为：{round(results["f1_score"], 3)}')