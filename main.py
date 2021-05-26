import time

import numpy as np
from matplotlib import pyplot as plt

from visualisation import plot_weapon_assignment
from wta_brute_force_solver import BruteForceSolver
from wta_genetic_solver import WTAGeneticSolver

np.random.seed(sum([ord(c) for c in '4142']))

W = 8
T = 4

V = np.random.uniform(low=1, high=10, size=(T))
K = np.random.uniform(low=0, high=1, size=(W, T))
n_chromosomes = 10
n_turns = 1000

genetic_solver = WTAGeneticSolver(W, T, V, K, n_chromosomes)

bf_solver = BruteForceSolver(W, T, V, K)

start = time.time()
p_val_min, p_min, P_values_arr = genetic_solver.solve(n_turns)
end = time.time()
print(f'Genetic: elapsed time: {end - start}')

start = time.time()
bf_val_min, bf_ass = bf_solver.solve()
end = time.time()
print(f'Brute force: elapsed time: {end - start}')

print(p_min == bf_ass)
print(bf_val_min, bf_ass)

plot_weapon_assignment(W, T, V, p_min)


def plot_P_values(P_values):
    plt.plot(np.arange(1, len(P_values) + 1), P_values)
    plt.title(f'W = {W}, T = {T}, start_chromosomes = {n_chromosomes}')
    plt.xlabel('iterations')
    plt.ylabel('cost function')
    plt.show()


plot_P_values(P_values_arr)
