# Selection - K Tournament Selection
# Crossover - One-Point Crossover
# Mutation - Bit-Wise Mutation
# Generational Replacement

''' Genetic Algorithm

pop <- initialize_population()
while (termination condition not satisfied)
{
    parents <- selection(pop)

    offsprings = list()
    for p1, p2 in parents
    {
        c <- crossover(p1, p2)
        c <- mutation(c)
        offsprings.append(c)
    }
    cost <- evaluate(offsprings)

    Replace population with new generation offsprings
}

'''

from numpy.random import randint
from numpy.random import rand

# Parameters
num_bits = 100                   # No. of bits in binary bitstring
pop_size = 40                   # Population size
cross_r = 0.1                   # Crossover rate
mut_r = 1.0/float(num_bits)     # Mutation rate
num_gen = 500                   # Maximum no. of generation

# Initial population
def generate_population(num_bits, pop_size):
    return [randint(0, 2, num_bits).tolist() for _ in range(pop_size)]

# Fitness of chromosome
def evaluate(chromosome):
    return sum(chromosome)

# Parent Selection - Tournament Selection
def selection(pop, scores, k = 3):
    tournament_ids = randint(0, len(pop), k)

    select_id = tournament_ids[0]
    for id in tournament_ids:
        if scores[select_id] < scores[id]:
            select_id = id

    return pop[select_id]

# Crossover - One-Point Crossover
def crossover(p1, p2, cross_r = 0.1):
    c1, c2 = p1.copy(), p2.copy()
    crossover_point = randint(1, num_bits-1)
    c1 = p1[:crossover_point] + p2[crossover_point:]
    c2 = p2[:crossover_point] + p1[crossover_point:]
    return [c1, c2]

# Mutation - Bit-Wise Mutation
def mutation(bitstring, mut_r):
    for i in range(len(bitstring)):
        if rand() < mut_r:
            bitstring[i] = 1 - bitstring[i]

# Genetic Algorithm
def genetic_algorithm(num_bits, pop_size, cross_r, mut_r, num_gen):
    # Genrerating intial population
    pop = generate_population(num_bits, pop_size)

    # Initializing best solution
    best_sol, best_sol_fitness = pop[0], evaluate(pop[0])

    gen = 0
    while gen < num_gen and best_sol_fitness != num_bits:
        # Evaluating chromosomes in popoulation
        fitness = [evaluate(chromosome) for chromosome in pop]

        # Select parents
        parents = [selection(pop, fitness) for _ in range(pop_size)]

        # Creating offsprings through genetic evolution
        offsprings = list()
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[i+1]
            for child in crossover(p1, p2, cross_r):
                mutation(child, mut_r)
                offsprings.append(child)
        
        # Replacing Population - Generational Replacement
        pop = offsprings

        # Evaluating chromosomes in popoulation
        fitness = [evaluate(chromosome) for chromosome in pop]

        # Best Solution
        for i in range(pop_size):
            if (fitness[i] > best_sol_fitness):
                best_sol, best_sol_fitness = pop[i], fitness[i]

        print(f"{gen+1} - Best Chromosome: {best_sol} | Fitness: {best_sol_fitness}")
        gen += 1

    return [best_sol, best_sol_fitness]

if __name__ == "__main__":
    best_sol, fitness = genetic_algorithm(num_bits, pop_size, cross_r, mut_r, num_gen)
    print("\nGenetic Alorithm Finished!")
    print(f"Best Chromosome: {best_sol} | Fitness: {fitness}")

