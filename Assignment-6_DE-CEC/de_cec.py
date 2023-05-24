import numpy as np
from numpy.random import uniform
from numpy.random import rand
from numpy.random import randint
import math
import cec_benchmark as cec

# Parameters
dim = 10
pop_size = 200
n_iter = 200
min_limit = 0
max_limit = 0

# Initial population
def generate_population():
    return [[uniform(min_limit, max_limit) for _ in range(dim)] for _ in range(pop_size)]  
    
def isBetter(X, Y):
    if math.fabs(evaluate(np.array(X))) < math.fabs(evaluate(np.array(Y))):
        return True
    else:
        return False

# Mutation
def mutation(pop, M_r = 0.2):
    best_sol = pop[0]
    best_sol_index = 0

    for i in range(pop_size):
        if isBetter(pop[i], best_sol):
            best_sol, best_sol_index = pop[i], i

    found_neighbour = False
    while not found_neighbour:
        KL = randint(0, len(pop), 2)
        if (best_sol_index != KL[0] != KL[1]):
            found_neighbour = True
    
    X = best_sol
    X_k = pop[KL[0]]
    X_l = pop[KL[1]]
    phi = rand()
    V = best_sol.copy()
    for i in range(len(best_sol)):
        V[i] = X[i] + phi*(X_k[i] - X_l[i]) if rand() < M_r else X[i]
    
    return V

# Crossover - Exponential Crossover
def crossover(target, mutant, C_r = 0.2):
    n = len(target)
    offspring = target.copy()
    j = randint(0, n)
    offspring[j] = mutant[j]

    p = 1
    while rand() < C_r and p < n:
        offspring[(j+p+1) % n] = mutant[(j+p+1) % n]
        p += 1
    
    return offspring

def DE():
    # Genrerating intial population
    pop = generate_population()

    best_sol = pop[0]
    best_sol_fitness = evaluate(np.array(best_sol))

    for iter in range(n_iter):
        for i in range(pop_size):
            X = pop[i]
            mutant = mutation(pop)
            trial = crossover(X, mutant)

            if isBetter(trial, X):
                pop[i] = trial
        
            if isBetter(pop[i], best_sol):
                best_sol, best_sol_fitness = pop[i], evaluate(np.array(pop[i]))
        
        print(f"{iter+1} - Best Solution: {best_sol} | Fitness: {best_sol_fitness}")
    
    return best_sol, best_sol_fitness

if __name__ == "__main__":
    print("Select Benchmark Function:\n\
          1. Griewank Function \n\
          2. Rastrigin Function \n\
          3. Rosenbrock Function \n\
          4. Ackley Function \n\
          5. Schwefel Function \n")
    
    func_no = int(input("Enter Function Number [1...5]: "))
    
    if func_no < 1 or func_no > 5:
        print("ERR, Invalid Selection!")
        exit(1)
    
    if func_no == 1:
        min_limit = -600
        max_limit = 600
        from cec_benchmark import griewank as evaluate
    elif func_no == 2:
        min_limit = -15
        max_limit = 15
        from cec_benchmark import rastrigin as evaluate
    elif func_no == 3:
        min_limit = -15
        max_limit = 15
        from cec_benchmark import rosenbrock as evaluate
    elif func_no == 4:
        min_limit = -32.768
        max_limit = 32.768
        from cec_benchmark import ackley as evaluate
    elif func_no == 5:
        min_limit = -500
        max_limit = 500
        from cec_benchmark import schwefel as evaluate
    
    best_sol, fitness = DE()
    print(f"Best Chromosome: {best_sol} | Fitness: {fitness}")

