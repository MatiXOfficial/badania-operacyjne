import numpy as np


class WTAGeneticSolver:

    def __init__(self, W: int, T: int, V: np.array, K: np.array, n_chromosomes: int, seed: int = None):
        """
        :param W: number of weapons
        :param T: number of targets
        :param V: target values, size=(T)
        :param K: target destruction probabilities, size=(W, T)
        :param n_chromosomes: number of chromosomes in each generation
        :param seed: seed for RNG
        """
        self.W = W
        self.T = T
        self.V = V
        self.K = K
        self.n_chromosomes = n_chromosomes
        if seed is not None:
            np.random.seed(seed)

        self.good_genes = np.argmax(K * V, axis=1)

    def solve(self, n_turns: int = 1000, report_interval: int = None):
        """
        Use the genetic algorithm to try to find the optimal solution.
        :param n_turns: Number of turns in the simulation
        :param report_interval: Interval between printed reports. If None, the interval is (10% * n_turns)
        :return: tuple: (found cost function, found solution (best chromosome), array of best cost function values in
                         each iteration)
        """
        if report_interval is None:
            report_interval = 0.1 * n_turns

        # initialise the population
        P = np.random.randint(low=0, high=self.T, size=(self.n_chromosomes, self.W))
        P_values = np.array([self._survival_fun(self._chromosome_to_X(a)) for a in P])
        self._print_report(0, P, P_values)
        P_values_arr = []
        for turn in range(1, n_turns + 1):
            P, P_values = self._deliver_next_generation(P, P_values)
            idx = np.argmin(P_values)
            P_values_arr.append(P_values[idx])
            if turn % report_interval == 0:
                self._print_report(turn, P, P_values)

        idx = np.argmin(P_values)
        return P_values[idx], P[idx], P_values_arr

    @staticmethod
    def _print_report(turn, P, P_values):
        idx = np.argmin(P_values)
        print(f'{turn:8}: {P_values[idx]:10.4f} <- {P[idx]}')

    def _survival_fun(self, X):
        return np.sum(self.V * np.prod(np.power(1 - self.K, X), axis=0))

    def _chromosome_to_X(self, chromosome):
        return np.eye(self.T)[chromosome]

    def _EX(self, A, B, mc):
        A, B = A.copy(), B.copy()
        for _ in range(mc):
            # find good genes that are the same in both parents
            inherited_good_genes = np.logical_and(A == B, A == self.good_genes)
            bad_genes = np.where(inherited_good_genes is False)[0]
            if len(bad_genes) <= 2:
                break
            gene1, gene2 = np.random.choice(bad_genes, 2, replace=False)
            A[gene1], A[gene2], B[gene1], B[gene2] = B[gene2], B[gene1], A[gene2], A[gene1]
        return A, B

    def _mutate(self, A, mm):
        indices_to_mutate = np.random.choice(np.arange(0, self.W), mm, replace=False)
        for idx in indices_to_mutate:
            A[idx] = np.random.randint(0, self.T)

    def _select(self, P, P_values, D):
        D_values = np.array([self._survival_fun(self._chromosome_to_X(a)) for a in D])
        all_values = np.concatenate((P_values, D_values))
        all_chromosomes = np.concatenate((P, D))
        indices = np.argpartition(all_values, self.n_chromosomes)[:self.n_chromosomes]
        new_P = np.take(all_chromosomes, indices, axis=0)
        new_values = np.take(all_values, indices)
        return new_P, new_values

    def _generate_offspring(self, P):
        D = []
        mc = min(int(np.random.gamma(1, 2)), self.W)
        mm = min(int(np.random.gamma(1, 2)), self.W)

        for i in range(0, self.n_chromosomes, 2):
            A, B = self._EX(P[i], P[i + 1], mc)
            D.append(A)
            D.append(B)
            self._mutate(D[i], mm)
            self._mutate(D[i + 1], mm)
        D = np.array(D)
        return D

    def _deliver_next_generation(self, P, P_values):
        D = self._generate_offspring(P)
        return self._select(P, P_values, D)
