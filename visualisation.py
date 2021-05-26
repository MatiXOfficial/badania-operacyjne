import numpy as np
from matplotlib import pyplot as plt


def plot_weapon_assignment(W, T, V, assignment):
    plot_range = 100.0
    weapons_x = np.array([1.0 for _ in range(W)])
    weapons_y = np.array([-1.0 * i * plot_range / (W - 1) for i in range(W)])
    weapons_size = np.array([100 * (100 / W) for _ in range(W)])
    targets_x = np.array([10.0 for _ in range(T)])
    targets_y = np.array([-1.0 * i * plot_range / (T - 1) for i in range(T)])
    target_max_value = np.max(V)
    targets_size = np.array([100 * V[i] * (100 / T) / target_max_value for i in range(T)])

    plt.figure(figsize=(20, 20))
    for i in range(len(assignment)):
        plt.plot([1.0, 10.0], [-1.0 * i * plot_range / (W - 1), -1.0 * assignment[i] * plot_range / (T - 1)], color='g')
    plt.scatter(weapons_x, weapons_y, s=weapons_size, color='r')
    plt.scatter(targets_x, targets_y, s=targets_size, color='b')
    plt.gca().axis('off')
    plt.show()
