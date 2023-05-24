import numpy as np
from numpy.random import choice
from numpy.random import rand
from numpy.random import permutation
import re
import math
import sys

# Parameters
kmax = 4
lmax = 4
num_city = None
n_iter = 500

nodes = {}

def import_nodes(filename):
    file = open(filename, 'r')

    for line in file:
        tokens = line.split()
        
        # print(tokens)
        city = tokens[0]
        if (re.match('^[0-9]*$', city)):
            try:
                x = int(tokens[1])
                y = int(tokens[2])
            except:
                x = int(float(tokens[1]))
                y = int(float(tokens[2]))
            nodes[int(city)] = {'x': x, 'y': y}

    file.close()

# Initial pSolution
def generate_solution(num_city):
    return list(permutation(num_city)+1)

def inverse(X):
    chromosome = X.copy()
    size = len(chromosome)
    mutation_points = choice(range(size), size=2, replace=False)
    start, end = min(mutation_points), max(mutation_points)

    mutate_chromosome = chromosome[:start] + chromosome[start:end+1][::-1] + chromosome[end+1:]
    return mutate_chromosome

def swap(X):
    chromosome = X.copy()
    size = len(chromosome)
    swap_points = choice(range(size), size=2, replace=False)

    chromosome[swap_points[0]], chromosome[swap_points[1]] =  chromosome[swap_points[1]], chromosome[swap_points[0]]
    return chromosome

def insert(X):
    chromosome = X.copy()
    size = len(chromosome)
    insert_points = choice(range(size), size=2, replace=False)
    i = insert_points[0]
    j, item_j = insert_points[1], chromosome[insert_points[1]]

    del chromosome[j]
    chromosome.insert(i, item_j)

    return chromosome

def three_opt(X):
    chromosome = X.copy()
    n = len(chromosome)
    insert_points = choice(range(n), size=2, replace=False)
    i, j = min(insert_points), max(insert_points)
    
    k = choice([k for k in range(n) if k != i and k != j])
    if k > j:
        a, b, c = chromosome[i:j+1], chromosome[j+1:k+1], chromosome[k+1:]
        if rand() < 0.5:
            chromosome[i:] = a + b + c
        else:
            chromosome[i:] = a + c + b
    elif i < k <= j:
        a, b, c = chromosome[i:k+1], chromosome[k+1:j+1], chromosome[j+1:]
        if rand() < 0.5:
            chromosome[i:] = a + b + c
        else:
            chromosome[i:] = a + c + b
    elif k <= i:
        a, b, c = chromosome[k:i+1], chromosome[i+1:j+1], chromosome[j+1:]
        if rand() < 0.5:
            chromosome[k:] = c + b + a
        else:
            chromosome[k:] = b + a + c
    
    return chromosome

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

def isBetter(X, Y):
    if math.fabs(evaluate(X)) < math.fabs(evaluate(Y)):
        return True
    else:
        return False

def shake(X, k):
    if k == 1:
        V = inverse(X)
    elif k == 2:
        V = swap(X)
    elif k == 3:
        V = insert(X)
    elif k == 4:
        V = three_opt(X)
    return V

def Neighbourhood_Change_Sequential(X, Xt, k):
    if isBetter(Xt, X):
        X = Xt
        k = 0
    else:
        k += 1
    return X, k

def VND(X, lmax):
    stop = False
    while not stop:
        l = 0
        Xt = X.copy()
        while l < lmax:
            neighbours = []
            for i in range(lmax):
                neighbours.append(shake(X, i+1))
            fitness = [evaluate(N) for N in neighbours]
            best_index = np.argmin(np.array(fitness))
            Xtt = neighbours[best_index]

            X, l = Neighbourhood_Change_Sequential(X, Xtt, l)
            # print(l)


        if isBetter(Xtt, Xt):
            stop = True
    
    return Xtt

def GVNS():
    X = generate_solution(num_city)
    iter = 0
    while iter < n_iter:
        k = 0
        while k < kmax:
            Xt = shake(X, k+1)
            Xtt = VND(Xt, lmax)
            X, k = Neighbourhood_Change_Sequential(X, Xtt, k)

        print(f"{iter+1} - Best Solution: {X} | Fitness: {evaluate(X)}")
        iter += 1

    return X, evaluate(X)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: <filename> missing!")
        exit(1)
    filename = sys.argv[1]
    import_nodes(filename)
    num_city = len(nodes)

    best_sol, fitness = GVNS()
    print(f"Best Solution: {best_sol} | Fitness: {fitness}")


