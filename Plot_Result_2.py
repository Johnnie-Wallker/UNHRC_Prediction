import matplotlib.pyplot as plt
import pandas as pd

# Data
accuracy = [
    0.671, 0.671, 0.683, 0.665, 0.662, 0.675, 0.660, 0.659, 0.653, 0.669,
    0.673, 0.676, 0.666, 0.667, 0.667, 0.672, 0.667, 0.672, 0.668, 0.655
]

recall = [
    0.503, 0.503, 0.521, 0.494, 0.489, 0.509, 0.487, 0.485, 0.477, 0.501,
    0.507, 0.511, 0.496, 0.497, 0.497, 0.504, 0.497, 0.504, 0.499, 0.479
]

accuracy_non_random = [
    0.691, 0.696, 0.693, 0.693, 0.689, 0.692, 0.693, 0.693, 0.698, 0.699,
    0.695, 0.692, 0.700, 0.696, 0.693, 0.693, 0.690, 0.698, 0.695, 0.689
]

recall_non_random = [
    0.534, 0.541, 0.537, 0.536, 0.531, 0.535, 0.537, 0.536, 0.545, 0.546,
    0.540, 0.535, 0.547, 0.541, 0.536, 0.536, 0.532, 0.545, 0.540, 0.531
]

combined_data = pd.DataFrame({
    'Accuracy': accuracy + accuracy_non_random,
    'Recall': recall + recall_non_random,
    'Group': ['Random']*20 + ['Non-Random']*20
})

# Plotting side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy boxplot
combined_data.boxplot(column='Accuracy', by='Group', ax=axes[0])
axes[0].set_title('Accuracy Comparison')
axes[0].set_xlabel('')
axes[0].set_ylabel('Accuracy')

# Recall boxplot
combined_data.boxplot(column='Recall', by='Group', ax=axes[1])
axes[1].set_title('Recall Comparison')
axes[1].set_xlabel('')
axes[1].set_ylabel('Recall')

plt.suptitle('')
plt.tight_layout()
plt.show()