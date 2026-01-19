import numpy as np
import matplotlib.pyplot as plt

def distillation_cost(n, m_x, k, p_physical, d):

    tiles = (1.5*(m_x + k) + 4)
    
    distillation_clock_cycles = (n - m_x)

    success_prob_for_k = (1 - p_physical)**n

    cost_function = (tiles * distillation_clock_cycles / k * success_prob_for_k) * d**3

    return cost_function

p_physical = 10**(-3)
d = 7

# find optimal n, m_x, and k
cost_list = []
params_list = []
for n in range(1, 50):
    for m_x in range(1, n):
        for k in range(1, n):
            cost = distillation_cost(n, m_x, k, p_physical, d)
            print(f"n: {n}, m_x: {m_x}, k: {k}, cost: {cost}")
            cost_list.append(cost)
            params_list.append((n, m_x, k))

# find the minimum cost
minimum_cost = min(cost_list)
minimum_cost_params = params_list[cost_list.index(minimum_cost)]
print(f"Minimum cost: {minimum_cost}, n: {minimum_cost_params[0]}, m_x: {minimum_cost_params[1]}, k: {minimum_cost_params[2]}")

# plot cost function against n, m_x, and k
plt.plot(cost_list)
plt.xlabel('n')
plt.ylabel('cost')
plt.title('Cost Function against n, m_x, and k')
plt.show()


