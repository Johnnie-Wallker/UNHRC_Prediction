import matplotlib.pyplot as plt
import numpy as np

# Data
methods = [
    'XGBoost', 'XGBoost (All Details)',
    'Large Model No Example', 'Large Model No Example (Edu)', 'Large Model No Example (Work)', 'Large Model No Example (All Details)',
    'Large Model With Example', 'Large Model With Example (Edu)', 'Large Model With Example (Work)', 'Large Model With Example (All Details)',
    'Large Model Summary Text', 'Large Model Summary Text (Edu)', 'Large Model Summary Text (Work)', 'Large Model Summary Text (All Details)',
    'Large Model With Example + Summary Text', 'Large Model With Example + Summary Text (Edu)',
    'Large Model With Example + Summary Text (Work)', 'Large Model With Example + Summary Text (All Details)'
]

accuracies = [
    0.689, 0.680,
    0.675, 0.663, 0.701, 0.689,
    0.684, 0.672, 0.716, 0.725,
    0.670, 0.680, 0.711, 0.701,
    0.682, 0.655, 0.713, 0.704
]

recalls = [
    0.310, 0.289,
    0.278, 0.251, 0.337, 0.310,
    0.299, 0.273, 0.369, 0.390,
    0.267, 0.289, 0.358, 0.337,
    0.294, 0.235, 0.364, 0.342
]

# Plotting
x = np.arange(len(methods))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
bars2 = ax.bar(x + width/2, recalls, width, label='Recall')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Methods')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison of Various Models and Configurations (Round 2)')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=90)
ax.legend()


# Attach a text label above each bar in *bars1* and *bars2*, displaying its height.
def autolabel(bars):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(bars1)
autolabel(bars2)

fig.tight_layout()

plt.show()
