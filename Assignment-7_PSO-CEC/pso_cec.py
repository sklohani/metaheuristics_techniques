import numpy as np
from numpy.random import uniform
from numpy.random import randint
from numpy.random import rand
import math
import cec_benchmark as cec

# Parameters
dim = 10
pop_size = 200
n_iter = 200
min_limit = 0
max_limit = 0
Vmin = 0
Vmax = 0
c1 = 2.0
c2 = 2.0
w_s = 0.9
w_e = 0.4
Tmax = n_iter

# Initial population
def generate_population():
    return [[uniform(min_limit, max_limit) for _ in range(dim)] for _ in range(pop_size)]

def isBetter(X, Y):
    if math.fabs(evaluate(np.array(X))) < math.fabs(evaluate(np.array(Y))):
        return True
    else:
        return False
    
def update_velocity(Xi, Vi, Pi, Pg, t):
    phi1 = rand()
    phi2 = rand()
    w = ((Tmax - t) * (w_s - w_e))/Tmax + w_e
    V_new = w*Vi + c1*phi1*(Pi - Xi) + c2*phi2*(Pg - Xi)

    if V_new < Vmin: V_new = Vmin
    elif V_new > Vmax: V_new = Vmax

    return V_new

def update_position(Xi, V_new):
    return Xi + V_new

def PSO():
    pop = generate_population()
    Pbest = pop.copy()
    Pbest_fitness = [evaluate(np.array(X)) for X in pop]
    Gbest_index = np.argmin(np.array(Pbest_fitness))
    Gbest_sol = Pbest[Gbest_index]
    V = [[0.0 for _ in range(dim)] for _ in range(pop_size)]

    best_sol = pop[0]
    best_sol_fitness = evaluate(np.array(best_sol))

    for iter in range(n_iter):
        for i in range(pop_size):

            X_sol = pop[i]
            Vi_sol = V[i]
            Pi_sol = Pbest[i]
            Pg_sol = Gbest_sol
            for d in range(dim):
                Vi_sol[d] = update_velocity(X_sol[d], Vi_sol[d], Pi_sol[d], Pg_sol[d], iter+1)
                X_sol[d] = update_position(X_sol[d], Vi_sol[d])

            if isBetter(X_sol, pop[i]):
                pop[i] = X_sol
                V[i] = Vi_sol
        
        Pbest_fitness = [evaluate(np.array(X)) for X in pop]
        Gbest_index = np.argmin(np.array(Pbest_fitness))
        Gbest_sol = pop[Gbest_index]
        best_sol, best_sol_fitness = Gbest_sol, evaluate(np.array(pop[Gbest_index]))
        
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
        Vmin = min_limit = -600
        Vmax = max_limit = 600
        from cec_benchmark import griewank as evaluate
    elif func_no == 2:
        Vmin = min_limit = -15
        Vmax = max_limit = 15
        from cec_benchmark import rastrigin as evaluate
    elif func_no == 3:
        Vmin = min_limit = -15
        Vmax = max_limit = 15
        from cec_benchmark import rosenbrock as evaluate
    elif func_no == 4:
        Vmin = min_limit = -32.768
        Vmax = max_limit = 32.768
        from cec_benchmark import ackley as evaluate
    elif func_no == 5:
        Vmin = min_limit = -500
        Vmax = max_limit = 500
        from cec_benchmark import schwefel as evaluate

    best_sol, fitness = PSO()
    print(f"Best Solution: {best_sol} | Fitness: {fitness}")

    