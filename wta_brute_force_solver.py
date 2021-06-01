import math

import numpy as np


class BruteForceSolver:
    def __init__(self, W: int, T: int, V: np.array, K: np.array):
        """
        :param W: number of weapons
        :param T: number of targets
        :param V: target values, size=(T)
        :param K: target destruction probabilities, size=(W, T)
        """
        self.W = W
        self.T = T
        self.V = V
        self.K = K
        self.D = [0 for _ in range(self.W)]
        self.Z = []
        self.min_p = math.inf
        self.best_d = [0 for _ in range(self.W)]

    @staticmethod
    def survival_fun(X, V, K):
        return np.sum(V * np.prod(np.power(1 - K, X), axis=0))

    @staticmethod
    def chromosome_to_X(chromosome, T):
        return np.eye(T)[chromosome]

    def select_every_target(self, D, i):
        if i >= self.W:
            self.Z.append(D.copy())
            return
        for j in range(0, self.T):
            D[i] = j
            self.select_every_target(D, i + 1)

    def solve(self):
        self.select_every_target(self.D, 0)
        Z_array = np.array(self.Z)
        P_values = np.array(
            [BruteForceSolver.survival_fun(BruteForceSolver.chromosome_to_X(a, self.T), self.V, self.K) for a in
             Z_array])
        idx = np.argmin(P_values)
        return P_values[idx], Z_array[idx]

    def check_survival_fun(self):
        p = BruteForceSolver.survival_fun(BruteForceSolver.chromosome_to_X(self.D, self.T), self.V, self.K)
        if p < self.min_p:
            self.min_p = p
            self.best_d = self.D.copy()

    def check_every_value(self, D, i):
        if i >= self.W:
            self.check_survival_fun()
            return
        for j in range(0, self.T):
            D[i] = j
            self.check_every_value(D, i + 1)

    def solve_2(self):
        self.check_every_value(self.D, 0)
        return self.min_p, self.best_d
