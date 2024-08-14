import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Result_Round1/result_DeepSeek_None.csv')
plt.figure(figsize=(10, 6))
plt.scatter(df['Group_Size'], df['Recall_DeepSeek'], color='b', marker='o')
plt.xlabel('Group Size')
plt.ylabel('Recall DeepSeek')
plt.title('Token Count vs Recall DeepSeek')
plt.grid(True)
plt.show()