from DeepSeek import run_deepseek
from openai import OpenAI

config = {
    'prompt_type': 'Train',
    'stage': 2,
    'shuffle': False,
    'detail': True,
    'vote_count': 1,
    'client': OpenAI(api_key="sk-a5ed383c9510411fa288cf6d2bd8b52d", base_url="https://api.deepseek.com"),
    'model': "deepseek-chat"
}

results = run_deepseek(config)
print(f'准确率为：{round(results["accuracy"], 3)} 召回率为：{round(results["f1_score"], 3)}')

# 简历筛选轮：
# 无样例（10次投票） 准确率为：0.685 召回率为：0.524
# 有样例（10次投票） 准确率为：0.683 召回率为：0.519
# 总结文本（10次投票） 准确率为：0.696 召回率为：0.541
# 无样例（20次投票） 准确率为：0.704 召回率为：0.552