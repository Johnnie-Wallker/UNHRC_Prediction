import matplotlib.pyplot as plt
import pandas as pd

accuracy = [
    0.656, 0.651, 0.653, 0.649, 0.652, 0.653, 0.650, 0.657, 0.653, 0.650,
    0.651, 0.656, 0.647, 0.649, 0.657, 0.653, 0.656, 0.652, 0.652, 0.656
]
recall = [
    0.480, 0.473, 0.477, 0.470, 0.474, 0.477, 0.472, 0.482, 0.477, 0.472,
    0.473, 0.480, 0.468, 0.470, 0.482, 0.477, 0.480, 0.475, 0.474, 0.480
]
accuracy_random = [
    0.639, 0.626, 0.631, 0.638, 0.631, 0.629, 0.638, 0.640, 0.624, 0.630,
    0.645, 0.630, 0.637, 0.637, 0.639, 0.633, 0.622, 0.638, 0.636, 0.631
]
recall_random = [
    0.455, 0.436, 0.442, 0.454, 0.444, 0.440, 0.454, 0.456, 0.432, 0.441,
    0.464, 0.441, 0.451, 0.451, 0.455, 0.446, 0.430, 0.454, 0.450, 0.444
]
accuracy_detail = [
    0.691, 0.696, 0.693, 0.693, 0.689, 0.692, 0.693, 0.693, 0.698, 0.699,
    0.695, 0.692, 0.700, 0.696, 0.693, 0.693, 0.690, 0.698, 0.695, 0.689
]
recall_detail = [
    0.534, 0.541, 0.537, 0.536, 0.531, 0.535, 0.537, 0.536, 0.545, 0.546,
    0.540, 0.535, 0.547, 0.541, 0.536, 0.536, 0.532, 0.545, 0.540, 0.531
]
accuracy_random_detail = [
    0.671, 0.671, 0.683, 0.665, 0.662, 0.675, 0.660, 0.659, 0.653, 0.669,
    0.673, 0.676, 0.666, 0.667, 0.667, 0.672, 0.667, 0.672, 0.668, 0.655
]
recall_random_detail = [
    0.503, 0.503, 0.521, 0.494, 0.489, 0.509, 0.487, 0.485, 0.477, 0.501,
    0.507, 0.511, 0.496, 0.497, 0.497, 0.504, 0.497, 0.504, 0.499, 0.479
]

combined_data = pd.DataFrame({
    'Accuracy': accuracy + accuracy_random + accuracy_detail + accuracy_random_detail,
    'Recall': recall + recall_random + recall_detail + recall_random_detail,
    'Group': ['DO+ND']*20 + ['DO+FD']*20 + ['RO+ND']*20 + ['RO+FD']*20
})


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

combined_data.boxplot(column='Accuracy', by='Group', ax=axes[0])
axes[0].set_title('Accuracy Comparison')
axes[0].set_xlabel('')
axes[0].set_ylabel('Accuracy')

combined_data.boxplot(column='Recall', by='Group', ax=axes[1])
axes[1].set_title('Recall Comparison')
axes[1].set_xlabel('')
axes[1].set_ylabel('Recall')

plt.suptitle('')
plt.tight_layout()
plt.show()