import matplotlib.pyplot as plt
import numpy as np

methods = [
    'XGBoost', 'XGBoost (All Details)', 'LLM', 'LLM (Edu)', 'LLM (Work)', 'LLM (All Details)',
    'LLM With Example', 'LLM With Example (Edu)', 'LLM With Example (Work)', 'LLM With Example (All Details)',
    'LLM With Summary', 'LLM With Summary (Edu)', 'LLM With Summary (Work)', 'LLM With Summary (All Details)',
    'LLM With Example + Summary', 'LLM With Example + Summary (Edu)',
    'LLM With Example + Summary (Work)', 'LLM With Example + Summary (All Details)'
]

accuracies_R1 = [
    0.683, 0.698, 0.651, 0.664, 0.705, 0.685, 0.647, 0.654, 0.680,
    0.683, 0.647, 0.648, 0.695, 0.696, 0.648, 0.668, 0.690, 0.704
]
recalls_R1 = [
    0.521, 0.544, 0.473, 0.493, 0.555, 0.524, 0.466, 0.476, 0.512,
    0.519, 0.468, 0.469, 0.540, 0.541, 0.469, 0.498, 0.531, 0.551
]
accuracies_R2 = [
    0.689, 0.680, 0.675, 0.663, 0.701, 0.689, 0.684, 0.672, 0.716,
    0.725, 0.670, 0.680, 0.711, 0.701, 0.682, 0.655, 0.713, 0.704
]
recalls_R2 = [
    0.310, 0.289, 0.278, 0.251, 0.337, 0.310, 0.299, 0.273, 0.369,
    0.390, 0.267, 0.289, 0.358, 0.337, 0.294, 0.235, 0.364, 0.342
]

x = np.arange(len(methods))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

# Plot for R1
bars1_R1 = ax1.bar(x - width / 2, accuracies_R1, width, label='Accuracy')
bars2_R1 = ax1.bar(x + width / 2, recalls_R1, width, label='Recall')
ax1.set_xlabel('Methods')
ax1.set_ylabel('Scores')
ax1.set_title('Performance Comparison of Various Models and Configurations (Round 1)')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=90)
ax1.legend()

# Plot for R2
bars1_R2 = ax2.bar(x - width / 2, accuracies_R2, width, label='Accuracy')
bars2_R2 = ax2.bar(x + width / 2, recalls_R2, width, label='Recall')
ax2.set_xlabel('Methods')
ax2.set_title('Performance Comparison of Various Models and Configurations (Round 2)')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=90)
ax2.legend()


def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(bars1_R1, ax1)
autolabel(bars2_R1, ax1)
autolabel(bars1_R2, ax2)
autolabel(bars2_R2, ax2)
fig.tight_layout()
plt.show()