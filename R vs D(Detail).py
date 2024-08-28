import matplotlib.pyplot as plt
import pandas as pd

accuracy = {
    'DO_ND': [
        0.656, 0.651, 0.653, 0.649, 0.652, 0.653, 0.650, 0.657, 0.653, 0.650,
        0.651, 0.656, 0.647, 0.649, 0.657, 0.653, 0.656, 0.652, 0.652, 0.656],
    'RO_ND': [
        0.639, 0.626, 0.631, 0.638, 0.631, 0.629, 0.638, 0.640, 0.624, 0.630,
        0.645, 0.630, 0.637, 0.637, 0.639, 0.633, 0.622, 0.638, 0.636, 0.631],
    'DO_EDU': [
        0.668, 0.662, 0.662, 0.663, 0.657, 0.664, 0.665, 0.668, 0.655, 0.677,
        0.667, 0.663, 0.661, 0.660, 0.670, 0.665, 0.666, 0.661, 0.665, 0.665],
    'RO_EDU': [
        0.653, 0.666, 0.648, 0.648, 0.651, 0.647, 0.650, 0.639, 0.655, 0.657,
        0.653, 0.657, 0.631, 0.647, 0.652, 0.635, 0.639, 0.643, 0.648, 0.642],
    'DO_WORK': [
        0.693, 0.692, 0.698, 0.690, 0.691, 0.695, 0.696, 0.690, 0.694, 0.697,
        0.695, 0.695, 0.698, 0.693, 0.693, 0.691, 0.699, 0.690, 0.687, 0.698],
    'RO_WORK': [
        0.668, 0.678, 0.670, 0.673, 0.669, 0.673, 0.665, 0.668, 0.681, 0.675,
        0.661, 0.658, 0.665, 0.672, 0.663, 0.675, 0.664, 0.668, 0.674, 0.657],
    'DO_FD': [
        0.691, 0.696, 0.693, 0.693, 0.689, 0.692, 0.693, 0.693, 0.698, 0.699,
        0.695, 0.692, 0.700, 0.696, 0.693, 0.693, 0.690, 0.698, 0.695, 0.689],
    'RO_FD': [
        0.671, 0.671, 0.683, 0.665, 0.662, 0.675, 0.660, 0.659, 0.653, 0.669,
        0.673, 0.676, 0.666, 0.667, 0.667, 0.672, 0.667, 0.672, 0.668, 0.655],
}
recall = {
    'DO_ND': [
        0.480, 0.473, 0.477, 0.470, 0.474, 0.477, 0.472, 0.482, 0.477, 0.472,
        0.473, 0.480, 0.468, 0.470, 0.482, 0.477, 0.480, 0.475, 0.474, 0.480],
    'RO_ND': [
        0.455, 0.436, 0.442, 0.454, 0.444, 0.440, 0.454, 0.456, 0.432, 0.441,
        0.464, 0.441, 0.451, 0.451, 0.455, 0.446, 0.430, 0.454, 0.450, 0.444],
    'DO_EDU': [
        0.499, 0.489, 0.489, 0.492, 0.483, 0.493, 0.494, 0.498, 0.479, 0.512,
        0.497, 0.492, 0.488, 0.487, 0.502, 0.494, 0.496, 0.488, 0.494, 0.494],
    'RO_EDU': [
        0.477, 0.496, 0.469, 0.469, 0.473, 0.466, 0.472, 0.454, 0.479, 0.483,
        0.477, 0.483, 0.444, 0.466, 0.475, 0.449, 0.455, 0.461, 0.469, 0.459],
    'DO_WORK': [
        0.536, 0.535, 0.544, 0.532, 0.534, 0.540, 0.541, 0.532, 0.539, 0.542,
        0.540, 0.540, 0.545, 0.536, 0.536, 0.534, 0.546, 0.532, 0.527, 0.544],
    'RO_WORK': [
        0.499, 0.513, 0.502, 0.506, 0.501, 0.506, 0.493, 0.499, 0.518, 0.509,
        0.486, 0.484, 0.494, 0.504, 0.492, 0.509, 0.493, 0.498, 0.508, 0.481],
    'DO_FD': [
        0.534, 0.541, 0.537, 0.536, 0.531, 0.535, 0.537, 0.536, 0.545, 0.546,
        0.540, 0.535, 0.547, 0.541, 0.536, 0.536, 0.532, 0.545, 0.540, 0.531],
    'RO_FD': [
        0.503, 0.503, 0.521, 0.494, 0.489, 0.509, 0.487, 0.485, 0.477, 0.501,
        0.507, 0.511, 0.496, 0.497, 0.497, 0.504, 0.497, 0.504, 0.499, 0.479],
}

df_accuracies = pd.DataFrame(accuracy)
df_recalls = pd.DataFrame(recall)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
df_accuracies.boxplot(ax=axes[0])
axes[0].set_title("Accuracy Distribution")
axes[0].set_ylabel("Accuracy")
df_recalls.boxplot(ax=axes[1])
axes[1].set_title("Recall Distribution")
axes[1].set_ylabel("Recall")
plt.tight_layout()
plt.show()
