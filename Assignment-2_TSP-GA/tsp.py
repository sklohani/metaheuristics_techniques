# TSP Graph fully connected
# Selection - Binary Tournament Selection (BTS)
# Crossover - Ordered Crossover (OX)
# Mutation - 2-Opt Mutation (Inversion Mutation)
# Steady State Evolution

''' Genetic Algorithm

pop <- initialize_population()
while (termination condition not satisfied)
{
    do 
    {
        p1 <- BTS()
        p2 <- BTS()
    } while (p1 == p2)

    c <- crossover(p1, p2)
    c <- mutation(c)
    cost <- evaluate(c)

    if (unique(c))
    {
        Insert c in pop replacing the worst chromosome
    }
}

'''

import sys
import re
import json
import math
import collections
import numpy as np
from numpy.random import permutation
from numpy.random import randint
from numpy.random import choice

# Parameters
num_city = None   # No. of cities
pop_size = 200    # Population size
cross_r = 0.1     # Crossover rate
mut_r = 0.05      # Mutation rate
num_gen = 1500     # Maximum no. of generation

nodes = {}

def import_nodes(filename):
    file = open(filename, 'r')

    for line in file:
        tokens = line.split()
        
        # print(tokens)
        city = tokens[0]
        if (re.match('^[0-9]*$', city)):
            x = int(tokens[1])
            y = int(tokens[2])
            nodes[int(city)] = {'x': x, 'y': y}

    # # Printing the nodes
    # print("::City Nodes::")
    # print(json.dumps(nodes, indent=2))
    file.close()

# Initial population
def generate_population(num_city, pop_size):
    return [permutation(num_city)+1 for _ in range(pop_size)]

# Fitness of chromosome
def evaluate(chromosome):
    fitness = 0
    for i in range(len(chromosome)):
        city1 = chromosome[i]
        if i != len(chromosome) - 1:
            city2 = chromosome[i+1]
        else:
            city2 = chromosome[0]

        # Pseudo Euclidian Distance
        xd = nodes[city1]['x'] - nodes[city2]['x']
        yd = nodes[city1]['y'] - nodes[city2]['y']
        r12 = math.sqrt((xd*xd + yd*yd) / 10.0)
        t12 = round(r12)

        if t12 < r12:
            fitness += t12 + 1
        else:
            fitness += t12
    return fitness

# Parent Selection - Binary Tournament Selection
def selection(pop, scores, k = 2):
    tournament_ids = randint(0, len(pop), k)

    if scores[tournament_ids[0]] > scores[tournament_ids[1]]:
        select_id = tournament_ids[1]
    else:
        select_id = tournament_ids[0]

    # select_id = tournament_ids[0]
    # for id in tournament_ids:
    #     if scores[id] < scores[select_id]:
    #         select_id = id

    return pop[select_id]

# Crossover - Ordered Crossover
def crossover(p1, p2, cross_r = 0.1):
    size = len(p1)
    c1, c2 = [None]*size, [None]*size
    crossover_points = choice(range(size), size=2, replace=False)
    start, end = min(crossover_points), max(crossover_points)

    # Genreting Child 1
    for i in range(start, end + 1, 1):
        c1[i] = p1[i]
    pointer = 0
    for i in range(size):
        if c1[i] is None:
            while p2[pointer] in c1:
                pointer += 1
            c1[i] = p2[pointer]
    
    # Genreting Child 2
    for i in range(start, end + 1, 1):
        c2[i] = p2[i]
    pointer = 0
    for i in range(size):
        if c2[i] is None:
            while p1[pointer] in c2:
                pointer += 1
            c2[i] = p1[pointer]
    
    if evaluate(c1) < evaluate(c2):
        return c1
    else:
        return c2

# Mutation - 2-Opt Mutation (Inversion Mutation)
def mutation(chromosome, mut_r = 0.05):
    size = len(chromosome)
    mutation_points = choice(range(size), size=2, replace=False)
    start, end = min(mutation_points), max(mutation_points)

    mutate_chromosome = chromosome[:start] + chromosome[start:end+1][::-1] + chromosome[end+1:]
    return mutate_chromosome

def local_search(chromosome):
    return None


# Unique Chromosome
def unique(chromosome, pop):
    u_flag = True
    for tour in pop:
        # if collections.Counter(tour) == collections.Counter(chromosome):
        #     u_flag = False
        if np.array_equal(np.array(tour), np.array(chromosome)):
            u_flag = False
    return u_flag

# Genetic Algorithm
def genetic_algorithm():
    # Genrerating intial population
    pop = generate_population(num_city, pop_size)

    gen = 0
    while gen < num_gen:
        # Evaluating chromosomes in population
        fitness = [evaluate(chromosome) for chromosome in pop]

        # Parent Selection
        p1 = None
        p2 = None
        while True:
            p1 = selection(pop, fitness)
            p2 = selection(pop, fitness)
            if not np.array_equal(np.array(p1), np.array(p2)):
                break
        
        # Crossover
        offspring = crossover(p1, p2)

        # Mutation
        offspring = mutation(offspring)

        # Checking for uniqueness
        if (unique(offspring, pop)):
            # Worst Solution
            worst_sol_fitness = 0
            worst_sol_index = 0
            for i in range(pop_size):
                if (fitness[i] > worst_sol_fitness):
                    worst_sol_index, worst_sol_fitness = i, fitness[i]
            
            # Replace offspring with worst chromosome in population - Steady State Evolution
            pop[worst_sol_index] = offspring

        # Best Solution
        best_sol_fitness = sys.maxsize
        for i in range(pop_size):
            if (fitness[i] < best_sol_fitness):
                best_sol, best_sol_fitness = pop[i], fitness[i]

        print(f"{gen+1} - Best Chromosome: {best_sol} | Fitness: {best_sol_fitness}")
        gen += 1
    
    return [best_sol, best_sol_fitness]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: <filename> missing!")
        exit(1)
    filename = sys.argv[1]
    import_nodes(filename)
    num_city = len(nodes)

    best_sol, fitness = genetic_algorithm()
    print("\nGenetic Alorithm Finished!")
    print(f"Best Chromosome: {best_sol} | Fitness: {fitness}")

