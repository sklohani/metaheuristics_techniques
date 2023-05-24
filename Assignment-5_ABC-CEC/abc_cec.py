import numpy as np
from numpy.random import uniform
from numpy.random import randint
from numpy.random import rand
import math
import cec_benchmark as cec

# Parameters
dim = 10
pop_size = 200
ne = pop_size//2
no = pop_size//2
n_iter = 500
min_limit = 0
max_limit = 0

# Initial population
def generate_population():
    return [[uniform(min_limit, max_limit) for _ in range(dim)] for _ in range(ne)]

def generate_solution():
    return [uniform(min_limit, max_limit) for _ in range(dim)]

# Binary Tournament Selection
def selection(pop, k = 2):
    scores = [evaluate(np.array(X)) for X in pop]
    tournament_ids = randint(0, len(pop), k)

    if scores[tournament_ids[0]] > scores[tournament_ids[1]]:
        select_id = tournament_ids[1]
    else:
        select_id = tournament_ids[0]

    return pop[select_id]

# Modified Neighbour Solution
def neighbour(X, X_k, M_r = 0.2):
    phi = rand()
    V = X.copy()
    for i in range(len(X)):
        V[i] = X[i] + phi*(X[i] - X_k[i]) if rand() < M_r else X[i]
    return V

# Determine neighbour solution
def determine_neighbour(X, pop):
    found_neighbour = False
    while not found_neighbour:
        k = randint(0, len(pop))
        X_k = pop[k]
        if not np.array_equal(np.array(X), np.array(X_k)):
            V = neighbour(X, X_k)
            found_neighbour = True
    return V

def isBetter(X, Y):
    if math.fabs(evaluate(np.array(X))) < math.fabs(evaluate(np.array(Y))):
        return True
    else:
        return False

def ABC():
    # Generate intial population
    pop = generate_population()

    # Best solution selection
    best_sol = selection(pop)
    best_sol_fitness = evaluate(np.array(best_sol))

    history = [0]*ne
    limit = 5
    for iter in range(n_iter):
        # Scout
        for i in range(ne):
            X = pop[i]

            # Determine neighbour solution
            V = determine_neighbour(X, pop)
            
            if isBetter(V, X):
                pop[i] = V
                history[i] = 0
            elif (history[i] >= limit):
                pop[i] = generate_solution()
                history[i] = 0
            else:
                history[i] += 1
            
            if isBetter(pop[i], best_sol):
                best_sol, best_sol_fitness = pop[i], evaluate(np.array(pop[i]))
        
        # Onlooker
        for i in range(no):
            X = selection(pop)
            V = determine_neighbour(X, pop)

            if isBetter(V, X):
                pop[i] = V
                history[i] = 0

                if isBetter(V, best_sol):
                    best_sol, best_sol_fitness = V, evaluate(np.array(V))
        
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

    best_sol, fitness = ABC()
    print(f"Best Solution: {best_sol} | Fitness: {fitness}")

    