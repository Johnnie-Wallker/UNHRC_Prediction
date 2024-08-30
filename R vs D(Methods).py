import matplotlib.pyplot as plt
import pandas as pd

accuracy = {
    'D': [
        0.662, 0.666, 0.662, 0.664, 0.668, 0.663, 0.668, 0.667, 0.664, 0.662,
        0.665, 0.662, 0.661, 0.665, 0.663, 0.662, 0.668, 0.664, 0.661, 0.666],
    'D_Rnd': [
        0.659, 0.678, 0.674, 0.662, 0.675, 0.669, 0.665, 0.653, 0.679, 0.658,
        0.646, 0.659, 0.673, 0.656, 0.678, 0.652, 0.660, 0.671, 0.664, 0.664],
    'T': [
        0.660, 0.663, 0.668, 0.667, 0.668, 0.662, 0.654, 0.655, 0.665, 0.665,
        0.665, 0.665, 0.668, 0.669, 0.662, 0.665, 0.663, 0.666, 0.665, 0.662],
    'T_Rnd': [
        0.659, 0.664, 0.671, 0.678, 0.667, 0.674, 0.671, 0.665, 0.678, 0.675,
        0.671, 0.660, 0.663, 0.662, 0.659, 0.667, 0.690, 0.681, 0.675, 0.665],
    'S': [
        0.664, 0.668, 0.665, 0.667, 0.669, 0.666, 0.664, 0.664, 0.663, 0.665,
        0.667, 0.667, 0.670, 0.668, 0.670, 0.663, 0.671, 0.663, 0.666, 0.662],
    'S_Rnd': [
        0.676, 0.669, 0.670, 0.663, 0.667, 0.660, 0.664, 0.677, 0.675, 0.669,
        0.657, 0.669, 0.662, 0.673, 0.652, 0.662, 0.664, 0.667, 0.681, 0.658],
    'T+S': [
        0.670, 0.673, 0.668, 0.670, 0.670, 0.671, 0.667, 0.675, 0.675, 0.673,
        0.673, 0.675, 0.678, 0.684, 0.674, 0.669, 0.673, 0.670, 0.673, 0.671],
    'T+S_Rnd': [
        0.672, 0.661, 0.676, 0.681, 0.692, 0.679, 0.681, 0.672, 0.671, 0.673,
        0.677, 0.682, 0.671, 0.679, 0.666, 0.666, 0.660, 0.674, 0.685, 0.670]
}

recall = {
    'D': [
        0.489, 0.496, 0.489, 0.492, 0.499, 0.492, 0.498, 0.497, 0.493, 0.489,
        0.494, 0.491, 0.488, 0.494, 0.492, 0.489, 0.498, 0.493, 0.488, 0.496],
    'D_Rnd': [
        0.485, 0.515, 0.508, 0.489, 0.509, 0.501, 0.494, 0.477, 0.516, 0.484,
        0.465, 0.485, 0.507, 0.480, 0.515, 0.475, 0.487, 0.503, 0.493, 0.493],
    'T': [
        0.484, 0.486, 0.492, 0.491, 0.493, 0.484, 0.474, 0.476, 0.490, 0.490,
        0.487, 0.490, 0.493, 0.495, 0.484, 0.492, 0.487, 0.490, 0.490, 0.484],
    'T_Rnd': [
        0.478, 0.489, 0.499, 0.511, 0.490, 0.505, 0.496, 0.493, 0.509, 0.506,
        0.499, 0.484, 0.488, 0.487, 0.480, 0.490, 0.528, 0.510, 0.504, 0.490],
    'S': [
        0.493, 0.499, 0.494, 0.497, 0.501, 0.496, 0.493, 0.493, 0.492, 0.494,
        0.497, 0.497, 0.502, 0.498, 0.502, 0.492, 0.503, 0.492, 0.496, 0.491],
    'S_Rnd': [
        0.511, 0.501, 0.502, 0.492, 0.497, 0.487, 0.493, 0.512, 0.509, 0.501,
        0.482, 0.501, 0.491, 0.507, 0.475, 0.489, 0.493, 0.497, 0.518, 0.484],
    'T+S': [
        0.500, 0.504, 0.496, 0.499, 0.498, 0.502, 0.495, 0.508, 0.508, 0.503,
        0.504, 0.507, 0.511, 0.519, 0.505, 0.498, 0.505, 0.500, 0.503, 0.500],
    'T+S_Rnd': [
        0.501, 0.488, 0.510, 0.517, 0.529, 0.512, 0.516, 0.503, 0.503, 0.505,
        0.507, 0.517, 0.501, 0.514, 0.495, 0.491, 0.483, 0.503, 0.517, 0.502]
}

df_recalls = pd.DataFrame(recall)
df_accuracies = pd.DataFrame(accuracy)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
df_accuracies.boxplot(ax=axes[0])
axes[0].set_title("Accuracy Distribution")
axes[0].set_ylabel("Accuracy")
df_recalls.boxplot(ax=axes[1])
axes[1].set_title("Recall Distribution")
axes[1].set_ylabel("Recall")
plt.tight_layout()
plt.show()