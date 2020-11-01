import numpy as np
from scipy.optimize import rosen
from scipy.spatial import distance_matrix

epsilon = np.finfo(float).eps
np.random.seed(0)


def create_population(fitness_function, lwr_bnd, upp_bnd, n, d):
    population = np.random.random((n, d))
    population = lwr_bnd + population * (upp_bnd - lwr_bnd)
    fitnesses = np.apply_along_axis(fitness_function, 1, population)
    return population, fitnesses


def compute_number_sparks_si(fitness, m, i):
    f_max = np.max(fitness)
    return m * ((f_max - fitness[i] + epsilon) /
                (sum(f_max - fitness) + epsilon))  # Eq1

def round_si(si, a, b, m):
    if si < a * m:
        si = int(round(a * m))
    elif si > b * m:
        si = int(round(b * m))
    else:
        si = int(round(si))
    return si

def compute_explosion_amplitude(fitness, A_hat, i):
    f_min = min(fitness)
    return (A_hat * ((fitness[i] - f_min + epsilon) /
                 (sum(fitness - f_min) + epsilon)))  # Eq2





def fa_iteration(fitness_function, pop, fit, lwr_bnd, upp_bnd, n,
                 d, m, m_hat, A_hat, a, b):
    
    # extract best individual
    x_star_idx = np.argmin(fit)
    x_star = pop[x_star_idx, :]
    x_star_fit = fit[x_star_idx]

    pop = np.delete(pop, x_star_idx, axis=0)
    fit = np.delete(fit, x_star_idx)

    # narrow down to n individuals with a distance matrix
    overall_distance = np.sum(distance_matrix(pop, pop), axis=0)

    p = np.divide(overall_distance, np.sum(overall_distance))

    # extract n-1 indexes using p
    p_indexes = np.random.choice(range(m - 1), n - 1, replace=False, p=p)

    # produces n fireworks
    fireworks = np.zeros(shape=(3, 3))
    fireworks[0, :] = x_star
    fireworks[1:n, :] = pop[p_indexes, :]
    fitness = np.zeros(3)
    fitness[0] = x_star_fit
    fitness[1:n] = fit[p_indexes]
    
    new_population = np.array(fireworks)

    # compute Si

    for i in range(n):
        si = compute_number_sparks_si(fitness, m, i)
        si = round_si(si, a, b, m)
        ai = compute_explosion_amplitude(fitness, A_hat, i)
        sparks_i = np.zeros((si, d))
        for spark in range(si):
            sparks_i[spark, :] = fireworks[i, :]
            z = round(d * np.random.random())
            z = np.random.choice(range(d), z, replace=False)
            h = ai * np.random.uniform(-1, 1)
            
            # linear displacement
            sparks_i[spark, z] = sparks_i[spark, z] + h

            # map sparks back to the search space
            idx = np.where(sparks_i[spark, :] < lwr_bnd)
            sparks_i[spark, idx] = lwr_bnd[idx]
            idx = np.where(sparks_i[spark, :] > upp_bnd)
            sparks_i[spark, idx] = upp_bnd[idx]

        new_population = np.concatenate((new_population, sparks_i), axis=0)

    # produce the m_hat gaussian sparks
    idx = np.random.choice(range(len(new_population)), m_hat, replace = True)
    gaussian_sparks = np.array(new_population[idx, :])
    
    for i in range(m_hat):
        z = round(d * np.random.random())
        z = np.random.choice(range(d), z, replace=False)
#            perform gaussian displacement
        g = np.random.normal(1, 1)
        gaussian_sparks[i, z] = gaussian_sparks[i, z] * g
        
#            map sparks back to the search space
        idx = np.where(gaussian_sparks[i, :] < lwr_bnd)
        gaussian_sparks[i, idx] = lwr_bnd[idx]
        idx = np.where(gaussian_sparks[i, :] > upp_bnd)
        gaussian_sparks[i, idx] = upp_bnd[idx]
   
    new_population = np.concatenate((gaussian_sparks, new_population), axis = 0)
    
    new_population = new_population[:m, :]
    
    new_fitnesses = np.apply_along_axis(fitness_function, 1, new_population)

    return new_population, new_fitnesses


def de_iteration(fitness_function, population, fitnesses, lwr_bnd, upp_bnd, m,
                 d, F, pc, strategy):
    new_population = np.zeros((m, d))

    if strategy == 1:
        idx = np.reshape(np.random.choice(range(m), m * 3), (m, 3))
    else:
        idx = np.reshape(np.random.choice(range(m), m * 2), (m, 2))
        best = np.repeat(np.argmin(fitnesses), m)
        idx = np.concatenate((idx, best[:, None]), axis=1)

    v = population[idx[:, 0], :] + F * (population[idx[:, 1], :] -
                                        population[idx[:, 2], :])

    for i in range(m):
        idx = np.where(v[i, :] < lwr_bnd)
        v[i, idx] = lwr_bnd[idx]
        idx = np.where(v[i, :] > upp_bnd)
        v[i, idx] = upp_bnd[idx]

    r = np.random.random((m, d))
    idx = r < pc
    new_population[idx] = v[idx]
    idx = np.logical_not(idx)
    new_population[idx] = population[idx]

    new_fitnesses = np.apply_along_axis(fitness_function, 1, new_population)
    idx = new_fitnesses < fitnesses
    fitnesses[idx] = new_fitnesses[idx]
    population[idx, :] = new_population[idx, :]

    return population, fitnesses


def hybrid(fitness_function, lwr_bnd, upp_bnd, F, m=50, n=5, d=30,
           iterations=500, m_hat=13, A_hat=40, a=0.04, b=0.8, pc=0.8, strategy=1):
    population, fitnesses = create_population(fitness_function, lwr_bnd,
                                              upp_bnd, m, d)
    for t in range(iterations):

        alg = np.random.random(1)

        if alg < 0.4:
            population, fitnesses = de_iteration(fitness_function, population,
                                                 fitnesses, lwr_bnd, upp_bnd, m, d, F, pc, strategy)
        else:
            population, fitnesses = fa_iteration(fitness_function, population,
                                                 fitnesses, lwr_bnd, upp_bnd, n, d, m, m_hat, A_hat, a, b)

    return population, iterations


dimensions = 30
np.random.seed(1)
lb = np.repeat(-5, dimensions)
ub = np.repeat(5, dimensions)

pop, fit = hybrid(rosen, lb, ub, 0.6)

todas = list()
for j in range(1000):
    pop, fit = create_population(rosen, -5, 5, 5, 3)
    sparks = list()
    for i in range(5):
        si = compute_number_sparks_si(fit, m, i)
        si = round_si(si, a, b, m)
        sparks.append(si)
    todas.append(sum(sparks))
todas = np.array(todas)
    