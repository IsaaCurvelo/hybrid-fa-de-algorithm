import numpy as np
from scipy.optimize import rosen
from scipy.spatial import distance_matrix

epsilon = np.finfo(float).eps

def create_population(fitness_function, lwr_bnd, upp_bnd, n, d):
    population = np.random.random((n, d))
    population = lwr_bnd + population * (upp_bnd - lwr_bnd)
    fitnesses = np.apply_along_axis(fitness_function, 1, population)
    return population, fitnesses


def fa_iteration(fitness_function, population, fitnesses, lwr_bnd, upp_bnd, n1,
                 n2, d, m, big_a_hat, a, b, mg):
    pass







def de_iteration(fitness_function, population, fitnesses, lwr_bnd, upp_bnd, n,
                                 d, F, pc, strategy):
    
    new_population = np.zeros((n, d))
    
    if strategy == 1:
            idx = np.reshape(np.random.choice(range(n), n * 3), (n, 3))
    else:
        idx = np.reshape(np.random.choice(range(n), n * 2), (n, 2))
        best = np.repeat(np.argmin(fitnesses), n)
        idx = np.concatenate((idx, best[:, None]), axis = 1)
            
    v = population[idx[:, 0], :] + F * (population[idx[:, 1], :] - 
                  population[idx[:, 2], :])

    for i in range(n):
        idx = np.where(v[i, :] < lwr_bnd)
        v[i, idx] = lwr_bnd[idx]
        idx = np.where(v[i, :] > upp_bnd)
        v[i, idx] = upp_bnd[idx]
    
    r = np.random.random((n, d))
    idx = r < pc
    new_population[idx] = v[idx]
    idx = np.logical_not(idx)
    new_population[idx] = population[idx]
    
    new_fitnesses = np.apply_along_axis(fitness_function, 1, new_population)
    idx = new_fitnesses < fitnesses
    fitnesses[idx] = new_fitnesses[idx]
    population[idx, :] = new_population[idx, :]

    return population, fitnesses







def hybrid(fitness_function, lwr_bnd, upp_bnd, F, n1 = 50, n2 = 5, d = 30, 
           iterations = 500, m = 50, big_a_hat = 40, a = 0.04, b = 0.8, mg = 5,
           pc = 0.8, strategy = 1):
    
    population, fitnesses = create_population(fitness_function, lwr_bnd, 
                                              upp_bnd, n1, d)
    for t in range(iterations):
        
        alg = np.random.random(1)
        
        if alg < 0.4:
            population, fitnesses = de_iteration(fitness_function, population,
                         fitnesses, lwr_bnd, upp_bnd, n1, d, F, pc, strategy)
        else:
            population, fitnesses = fa_iteration(fitness_function, population,
             fitnesses, lwr_bnd, upp_bnd, n1, n2, d, m, big_a_hat, a, b, mg)
            
    return population, iterations
