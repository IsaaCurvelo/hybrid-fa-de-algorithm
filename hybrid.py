import numpy as np
from scipy.optimize import rosen
from scipy.spatial import distance_matrix

epsilon = np.finfo(float).eps


def create_population(fitness_function, lwr_bnd, upp_bnd, n, d):
    population = np.random.random((n, d))
    population = lwr_bnd + population * (upp_bnd - lwr_bnd)
    fitnesses = np.apply_along_axis(fitness_function, 1, population)
    return population, fitnesses


def compute_number_sparks_si(fitness, m, i):
    f_max = np.max(fitness)
    return m * ((f_max - fitness[i] + epsilon) /
                (sum(f_max - fitness) + epsilon))  # Eq1


def compute_explosion_amplitude(fitness, A, i):
    f_min = min(fitness)
    return (A * ((fitness[i] - f_min + epsilon) /
                 (sum(fitness - f_min) + epsilon)))  # Eq2


def round_si(si, a, b, number_sparks):
    if si < a * number_sparks:
        si = round(a * number_sparks)
    elif si > b * number_sparks:
        si = round(b * number_sparks)
    else:
        si = round(si)
    return si


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

    # extract n-1 indexes with p
    p_indexes = np.random.choice(range(m - 1), n - 1, replace=False, p=p)

    # produce the m_hat gaussian sparks

    # produce the remaining ones up to m

    return 1, 2


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
