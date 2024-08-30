import matplotlib.pyplot as plt
import numpy as np


methods = [
    'XGBoost', 'XGBoost(Edu)', 'XGBoost(Work)', 'XGBoost(All Details)',
    'LLM', 'LLM(Edu)', 'LLM(Work)', 'LLM(All Details)',
    'LLM Train', 'LLM Train(Edu)', 'LLM Train(Work)', 'LLM Train(All Details)',
    'LLM Summary', 'LLM Summary(Edu)', 'LLM Summary(Work)', 'LLM Summary(All Details)',
    'LLM Train+Summary', 'LLM Train+Summary(Edu)', 'LLM Train+Summary(Work)', 'LLM Train+Summary(All Details)'
]
accuracies_R1 = [
    0.683, 0.681, 0.679, 0.698,
    0.651, 0.664, 0.705, 0.685,
    0.647, 0.654, 0.680, 0.683,
    0.647, 0.648, 0.695, 0.696,
    0.648, 0.668, 0.690, 0.704
]
recalls_R1 = [
    0.521, 0.518, 0.516, 0.544,
    0.473, 0.493, 0.555, 0.524,
    0.466, 0.476, 0.512, 0.519,
    0.468, 0.469, 0.540, 0.541,
    0.469, 0.498, 0.531, 0.551
]
accuracies_R2 = [
    0.689, 0.692, 0.670, 0.680,
    0.675, 0.663, 0.701, 0.689,
    0.684, 0.672, 0.716, 0.725,
    0.670, 0.680, 0.711, 0.701,
    0.682, 0.655, 0.713, 0.704
]
recalls_R2 = [
    0.310, 0.316, 0.267, 0.289,
    0.278, 0.251, 0.337, 0.310,
    0.299, 0.273, 0.369, 0.390,
    0.267, 0.289, 0.358, 0.337,
    0.294, 0.235, 0.364, 0.342
]


def autolabel(bars, ax):
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.annotate(f'{width:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center')


y = np.arange(len(methods))
height = 0.35
fig1, ax1 = plt.subplots(figsize=(14, 16))
bars1_R1 = ax1.barh(y - height / 2, accuracies_R1, height, label='Accuracy')
bars2_R1 = ax1.barh(y + height / 2, recalls_R1, height, label='Recall')
ax1.set_ylabel('Methods')
ax1.set_xlabel('Scores')
ax1.set_title('Performance Comparison of Various Models and Configurations (Round 1)')
ax1.set_yticks(y)
ax1.set_yticklabels(methods)
ax1.legend()
autolabel(bars1_R1, ax1)
autolabel(bars2_R1, ax1)
fig1.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(14, 16))
bars1_R2 = ax2.barh(y - height / 2, accuracies_R2, height, label='Accuracy')
bars2_R2 = ax2.barh(y + height / 2, recalls_R2, height, label='Recall')
ax2.set_ylabel('Methods')
ax2.set_xlabel('Scores')
ax2.set_title('Performance Comparison of Various Models and Configurations (Round 2)')
ax2.set_yticks(y)
ax2.set_yticklabels(methods)
ax2.legend()
autolabel(bars1_R2, ax2)
autolabel(bars2_R2, ax2)
fig2.tight_layout()
plt.show()