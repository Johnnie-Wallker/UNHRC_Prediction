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
    0.683, 0.698,
    0.651, 0.664, 0.705, 0.685,
    0.647, 0.654, 0.680, 0.683,
    0.647, 0.648, 0.695, 0.696,
    0.648, 0.668, 0.690, 0.704
]

recalls = [
    0.521, 0.544,
    0.473, 0.493, 0.555, 0.524,
    0.466, 0.476, 0.512, 0.519,
    0.468, 0.469, 0.540, 0.541,
    0.469, 0.498, 0.531, 0.551
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
ax.set_title('Performance Comparison of Various Models and Configurations(Round 1)')
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
