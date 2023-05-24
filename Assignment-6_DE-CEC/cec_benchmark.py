import numpy as np
import math

# Griewank Function
# Global minima value = 0
# Initialization range = [-600, 600]
def griewank(X):
    t1 = 0
    t2 = 1
    for i in range(len(X)):
        t1 += X[i]*X[i]
        t2 *= math.cos(X[i]/(i+1))
    return t1/4000 - t2 + 1

# Rastrigin Function
# Global minima value = 0
# Initialization range = [-15, 15]
def rastrigin(X):
    return np.sum(X*X - 10*np.cos(2*math.pi*X) + 10)

# Rosenbrock Function
# Global minima value = 0
# Initialization range = [-15, 15]
def rosenbrock(X):
    res = 0
    for i in range(len(X)-1):
        res += 100*(X[i]*X[i] - X[i+1])**2 + (1 - X[i])**2
    return res

# Ackley Function
# Global minima value = 0
# Initialization range = [-32.768, 32.768]
def ackley(X):
    t1 = -0.2 * np.sqrt(np.mean(X**2))
    t2 = np.mean(np.cos(2*math.pi*X))
    # return 20 + math.e - 20*math.exp(t1) - math.exp(t2)
    return 20 + math.e - 20*(math.e**t1) - math.e**t2

# Schwefel Function
# Global minima value = 0
# Initialization range = [-500, 500]
def schwefel(X):
    t1 = 0
    for i in range(len(X)):
        t1 += -1 * X[i] * math.sin(math.sqrt(math.fabs(X[i])))
    return len(X) * 418.9829 * t1

