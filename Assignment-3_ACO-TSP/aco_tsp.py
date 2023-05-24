import sys
import re
import math
import numpy as np
from numpy.random import permutation
from numpy.random import randint
from numpy.random import choice

# Parameters
n_ants = 30
n_iter = 200
alpha = 1
beta = 2
rho = 0.5
Q = 1

nodes = []

def import_nodes(filename):
    file = open(filename, 'r')

    for line in file:
        tokens = line.split()

        city = tokens[0]
        if (re.match('^[0-9]*$', city)):
            x = int(tokens[1])
            y = int(tokens[2])
            nodes.append([x, y])
    file.close()

# Pseudo Euclidian Distance
def distance(node1, node2):
    xd = node1[0] - node2[0]
    yd = node1[1] - node2[1]
    r12 = math.sqrt((xd*xd + yd*yd)/10.0)
    t12 = round(r12)
    if t12 < r12:
        fitness = t12 + 1
    else:
        fitness = t12
    return fitness

def ACO():
    n_nodes = len(nodes)
    pheromone = np.ones((n_nodes, n_nodes))
    best_solution = None
    best_solution_fitness = np.inf

    for iter in range(n_iter):
        paths = []
        path_fitness = []

        for ant in range(n_ants):
            visited = [False]*n_nodes
            current_node = randint(n_nodes)
            visited[current_node] = True
            path = [current_node]
            path_fit = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                prob = np.zeros(len(unvisited))

                for i, unvisited_node in enumerate(unvisited):
                    prob[i] = pheromone[current_node, unvisited_node]**alpha / distance(nodes[current_node], nodes[unvisited_node])**beta

                prob /= np.sum(prob)

                next_node = choice(unvisited, p = prob)
                path.append(next_node)
                path_fit += distance(nodes[current_node], nodes[next_node])
                visited[next_node] = True
                current_node = next_node
            
            paths.append(path)
            path_fitness.append(path_fit)

            if path_fit < best_solution_fitness:
                best_solution = path
                best_solution_fitness = path_fit
            
        print(f"{iter+1} - Best Solution: {best_solution} | Fitness: {best_solution_fitness}")

        pheromone *= rho

        for path, path_fit in zip(paths, path_fitness):
            for i in range(n_nodes-1):
                pheromone[path[i], path[i+1]] += Q/path_fit
            pheromone[path[-1], path[0]] + Q/path_fit
    
    return best_solution, best_solution_fitness

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERR: <input-file> missing!")
        exit(1)

    filename = sys.argv[1]
    import_nodes(filename)

    best_solution, best_solution_fitness = ACO()
    print(f"Best Solution: {best_solution} | Fitness: {best_solution_fitness}")


    