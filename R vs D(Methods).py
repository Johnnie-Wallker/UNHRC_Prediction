import matplotlib.pyplot as plt
import pandas as pd

accuracy = {
    'D_DO': [
        0.693, 0.689, 0.688, 0.689, 0.690, 0.696, 0.695, 0.695, 0.693, 0.688,
        0.688, 0.693, 0.693, 0.693, 0.690, 0.690, 0.688, 0.690, 0.697, 0.689],
    'D_RO': [
        0.666, 0.661, 0.673, 0.653, 0.665, 0.657, 0.674, 0.660, 0.670, 0.667,
        0.673, 0.662, 0.678, 0.673, 0.663, 0.681, 0.659, 0.669, 0.667, 0.678],
    'T_DO': [
        0.685, 0.694, 0.691, 0.692, 0.687, 0.689, 0.678, 0.682, 0.684, 0.686,
        0.694, 0.689, 0.688, 0.683, 0.686, 0.679, 0.685, 0.689, 0.686, 0.691],
    'T_RO': [
        0.660, 0.667, 0.653, 0.674, 0.656, 0.660, 0.671, 0.660, 0.662, 0.658,
        0.663, 0.663, 0.655, 0.663, 0.655, 0.657, 0.657, 0.654, 0.657, 0.670],
    'S_DO': [
        0.688, 0.690, 0.684, 0.688, 0.688, 0.689, 0.688, 0.686, 0.692, 0.691,
        0.683, 0.688, 0.697, 0.686, 0.688, 0.693, 0.693, 0.688, 0.691, 0.695],
    'S_RO': [
        0.652, 0.673, 0.664, 0.662, 0.672, 0.676, 0.665, 0.656, 0.678, 0.669,
        0.669, 0.667, 0.673, 0.658, 0.670, 0.675, 0.674, 0.672, 0.657, 0.661],
    'T+S_DO': [
        0.684, 0.682, 0.683, 0.683, 0.687, 0.679, 0.682, 0.686, 0.686, 0.688,
        0.683, 0.686, 0.690, 0.681, 0.691, 0.691, 0.683, 0.677, 0.686, 0.684],
    'T+S_RO': [
        0.668, 0.657, 0.672, 0.671, 0.674, 0.662, 0.680, 0.660, 0.669, 0.664,
        0.672, 0.668, 0.669, 0.691, 0.677, 0.659, 0.677, 0.658, 0.670, 0.661]
}

recall = {
    'D_DO': [
        0.693, 0.694, 0.693, 0.695, 0.690, 0.694, 0.689, 0.691, 0.696, 0.688,
        0.690, 0.693, 0.692, 0.689, 0.691, 0.688, 0.696, 0.688, 0.690, 0.689],
    'D_RO': [
        0.671, 0.667, 0.661, 0.662, 0.668, 0.666, 0.661, 0.674, 0.663, 0.661,
        0.678, 0.675, 0.663, 0.666, 0.662, 0.667, 0.662, 0.673, 0.678, 0.674],
    'T_DO': [
        0.687, 0.685, 0.683, 0.683, 0.685, 0.691, 0.686, 0.691, 0.692, 0.692,
        0.689, 0.682, 0.688, 0.690, 0.682, 0.690, 0.686, 0.687, 0.688, 0.691],
    'T_RO': [
        0.668, 0.656, 0.667, 0.659, 0.657, 0.669, 0.662, 0.665, 0.663, 0.670,
        0.655, 0.660, 0.674, 0.668, 0.667, 0.659, 0.666, 0.654, 0.664, 0.664],
    'S_DO': [
        0.689, 0.693, 0.688, 0.688, 0.692, 0.694, 0.691, 0.693, 0.688, 0.686,
        0.688, 0.691, 0.694, 0.689, 0.690, 0.688, 0.688, 0.688, 0.689, 0.686],
    'S_RO': [
        0.678, 0.656, 0.661, 0.667, 0.667, 0.673, 0.658, 0.667, 0.670, 0.669,
        0.666, 0.669, 0.661, 0.669, 0.672, 0.665, 0.663, 0.670, 0.675, 0.672],
    'T+S_DO': [
        0.679, 0.688, 0.684, 0.686, 0.683, 0.682, 0.682, 0.685, 0.686, 0.683,
        0.683, 0.682, 0.683, 0.682, 0.683, 0.683, 0.688, 0.691, 0.681, 0.688],
    'T+S_RO': [
        0.655, 0.659, 0.672, 0.669, 0.664, 0.674, 0.667, 0.659, 0.660, 0.668,
        0.667, 0.677, 0.664, 0.657, 0.677, 0.674, 0.671, 0.670, 0.666, 0.661]
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