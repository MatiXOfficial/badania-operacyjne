import numpy as np
from numpy import random
 
seed = sum([ord(c) for c in 'And I will strike down upon thee with great vengeance and furious anger'])
np.random.seed(seed)

##### model config #####
# Weapon number
W = 6
# Target number
T = 4
# Target values
V = np.random.uniform(low=1, high=10, size=(T))
# Probability of target's destruction
K = np.random.uniform(low=0, high=1, size=(W, T))

##### GA config #####
# number of chromosomes
start_chromosomes = 20 * 2 + 1 * 3 + 2 * 2
# population
P = np.random.randint(low=0, high=T, size=(start_chromosomes, W))

def survival_fun(X, V, K):
    return np.sum(V * np.prod(np.power(1 - K, X), axis=0))
    
def chromosome_to_X(chromosome, T):
    return np.eye(T)[chromosome]

print(chromosome_to_X(P[0], T))
print(np.power(1 - K, chromosome_to_X(P[0], T)))
print(np.prod( np.power(1 - K, chromosome_to_X(P[0], T)), axis=0))
print(survival_fun(chromosome_to_X(P[0], T), V, K))