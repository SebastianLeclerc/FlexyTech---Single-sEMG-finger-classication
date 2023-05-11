import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from functools import partial

def sliding_window(data, window_size, overlap):
    """
    Applies a sliding window with overlap to the input data.

    Parameters:
    data (ndarray): input data
    window_size (int): size of sliding window
    overlap (int): overlap between consecutive windows

    Returns:
    ndarray: 2D array with sliding windows of shape (n_windows, window_size)
    """
    n_samples = data.shape[0]
    n_windows = int(np.floor((n_samples - window_size) / overlap) + 1)
    windows = np.zeros((n_windows, window_size))

    for i in range(n_windows):
        start = i * overlap
        end = start + window_size
        windows[i] = data[start:end]

    return windows

def pca_fitness(X, n_components, fitness_func):
    """
    Applies PCA to the input data and returns the fitness score.

    Parameters:
    X (ndarray): input data
    n_components (int): number of PCA components to use
    fitness_func (callable): function that calculates the fitness score

    Returns:
    float: fitness score
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return fitness_func(X_pca)

def genetic_algorithm(X, fitness_func, pop_size=50, n_generations=100, window_bounds=(50, 500), overlap_bounds=(1, 100)):
    """
    Uses a genetic algorithm to optimize the window and overlap sizes for a sliding window.

    Parameters:
    X (ndarray): input data
    fitness_func (callable): function that calculates the fitness score
    pop_size (int): population size for genetic algorithm
    n_generations (int): number of generations for genetic algorithm
    window_bounds (tuple): bounds for window size
    overlap_bounds (tuple): bounds for overlap size

    Returns:
    tuple: best window and overlap sizes found
    """
    def population_initializer():
        return (np.random.randint(window_bounds[0], window_bounds[1] + 1),
                np.random.randint(overlap_bounds[0], overlap_bounds[1] + 1))

    def fitness_wrapper(population, fitness_func, X):
        return [fitness_func(sliding_window(X, window_size, overlap)) for window_size, overlap in population]

    def selection(population, fitness_scores, pop_size):
        fitness_probs = fitness_scores / np.sum(fitness_scores)
        selected_indices = np.random.choice(np.arange(len(population)), size=pop_size, replace=True, p=fitness_probs)
        return [population[i] for i in selected_indices]

    def crossover(parents, n_offsprings):
        offsprings = []
        for i in range(n_offsprings):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            offspring = (parent1[0], parent2[1])
            offsprings.append(offspring)
        return offsprings

    def mutation(population, mutation_prob):
        mutated_population = []
        for individual in population:
            if np.random.random() < mutation_prob:
                mutated_window_size = np.random.randint(window_bounds[0], window_bounds[1] + 1)
                mutated_overlap = np.random.randint(overlap_bounds[0], overlap_bounds[1] + 1)
                mutated_individual = (mutated_window_size, mutated_overlap)
            else:
                mutated_individual = individual
            mutated_population.append(mutated_individual)
        return mutated_population

    population = [population_initializer() for _ in range(pop_size)]
    fitness_scores = fitness_wrapper(population, fitness_func, X)
    best_fitness = np.max(fitness_scores)
    best_individual = population[np.argmax(fitness_scores)]

    for i in range(n_generations):
        selected_population = selection(population, fitness_scores, pop_size)
        offsprings = crossover(selected_population, pop_size - len(selected_population))
        mutated_population = mutation(selected_population + offsprings, mutation_prob=0.05)
        fitness_scores = fitness_wrapper(mutated_population, fitness_func, X)
        best_index = np.argmax(fitness_scores)
        if fitness_scores[best_index] > best_fitness:
            best_fitness = fitness_scores[best_index]
            best_individual = mutated_population[best_index]
        population = mutated_population

    return best_individual
