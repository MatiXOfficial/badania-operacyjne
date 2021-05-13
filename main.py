import numpy as np
from matplotlib import pyplot as plt

seed = sum([ord(c) for c in 'Tuturururuuttuutut'])
np.random.seed(seed)

##### model config #####
# Weapon number
W = 50
# Target number
T = 40
# Target values
V = np.random.uniform(low=1, high=10, size=(T))
# Probability of target's destruction
K = np.random.uniform(low=0, high=1, size=(W, T))

np.random.seed()

##### GA config #####
# number of chromosomes
start_chromosomes = 50

def survival_fun(X, V, K):
    return np.sum(V * np.prod(np.power(1 - K, X), axis=0))
    
def chromosome_to_X(chromosome, T):
    return np.eye(T)[chromosome]

good_genes = np.argmax(K * V, axis=1)

def EX(A, B, mc):   # crossover
    A, B = A.copy(), B.copy()
    for _ in range(mc):
        inherited_good_genes = np.logical_and(A == B, A == good_genes)  # find good genes that are the same in both parents
        bad_genes = np.where(inherited_good_genes==False)[0]
        if len(bad_genes) <= 2 :
            break
        gene1, gene2 = np.random.choice(bad_genes, 2, replace=False)
        A[gene1], A[gene2], B[gene1], B[gene2] = B[gene2], B[gene1], A[gene2], A[gene1]
    return A, B

def mutate(A, mm):
    indices_to_mutate = np.random.choice(np.arange(0, W), mm, replace=False)
    for idx in indices_to_mutate:
        A[idx] = np.random.randint(0, T)

def select(P, P_values, D):
    D_values = np.array([survival_fun(chromosome_to_X(a, T), V, K) for a in D])
    all_values = np.concatenate((P_values, D_values))
    all_chromosomes = np.concatenate((P, D))
    indices = np.argpartition(all_values, start_chromosomes)[:start_chromosomes]
    new_P = np.take(all_chromosomes, indices, axis=0)
    new_values = np.take(all_values, indices)
    return new_P, new_values
    
def generate_offspring(P):
    D = []
    mc = min(int(np.random.gamma(1, 2)), W//4)
    mm = min(int(np.random.gamma(1, 2)), W//4)
    # mc = 3
    # mm = 2
    for i in range(0, start_chromosomes, 2):
        A, B = EX(P[i], P[i+1], mc)
        D.append(A)
        D.append(B)
        mutate(D[i], mm)
        mutate(D[i+1], mm)
    D = np.array(D)
    return D

def next_generation(P, P_values):
    D = generate_offspring(P)
    return select(P, P_values, D)

def simulate_world(turns):
    P = np.random.randint(low=0, high=T, size=(start_chromosomes, W))
    P_values = np.array([survival_fun(chromosome_to_X(a, T), V, K) for a in P])
    idx = np.argmin(P_values)
    print(P_values[idx], P[idx])
    P_values_arr = []
    for _ in range(turns):
        P, P_values = next_generation(P, P_values)
        idx = np.argmin(P_values)
        P_values_arr.append(P_values[idx])

    idx = np.argmin(P_values)
    return P_values[idx], P[idx], P_values_arr 

p_val_min, p_min, P_values_arr = simulate_world(10000)
print(p_val_min, p_min)

def plot_P_values(P_values):
    plt.plot(np.arange(1, len(P_values) + 1), P_values)
    plt.title(f'W = {W}, T = {T}, start_chromosomes = {start_chromosomes}')
    plt.xlabel('iterations')
    plt.ylabel('cost function')
    plt.show()

plot_P_values(P_values_arr)