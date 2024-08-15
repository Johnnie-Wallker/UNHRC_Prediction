import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('Result_Round1/result_DeepSeek_None_SmallGroup.csv')
df2 = pd.read_csv('Result_Round1/result_DeepSeek_None.csv')
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(df1['Group_Size'], df1['Recall_DeepSeek'], color='b', marker='o')
plt.xlabel('Group Size')
plt.ylabel('Recall DeepSeek')
plt.title('Token Count vs Recall DeepSeek (Small Group)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df2['Group_Size'], df2['Recall_DeepSeek'], color='r', marker='o')
plt.xlabel('Group Size')
plt.ylabel('Recall DeepSeek')
plt.title('Token Count vs Recall DeepSeek (Full Group)')
plt.grid(True)

plt.tight_layout()
plt.show()