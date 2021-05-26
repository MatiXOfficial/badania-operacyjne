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
        P_values = np.array([BruteForceSolver.survival_fun(BruteForceSolver.chromosome_to_X(a, self.T), self.V, self.K) for a in Z_array])
        idx = np.argmin(P_values)
        return P_values[idx], Z_array[idx]

