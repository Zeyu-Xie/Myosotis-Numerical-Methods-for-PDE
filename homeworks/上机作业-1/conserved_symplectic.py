import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

T = 10
dt = 0.01
steps = int(T / dt)

with open(os.path.join(os.path.dirname(__file__), "config.txt"), "r") as file:
    T = float(file.readline())
    dt = float(file.readline())
    steps = int(T / dt)

def f(t, y):
    p, q = y
    return [-q, p]


def hamiltonian(t, y):
    p, q = y
    return p**2 + q**2


# Solving using solve_ivp with 'RK45' method
sol = solve_ivp(f, [0, T], [0, 1], method='RK45',
                t_eval=np.linspace(0, T, steps))

# Extracting results
time = sol.t
p_values = sol.y[0]
q_values = sol.y[1]

# Computing conserved quantity
conserved_symplectic = (sol.y[0]**2 + sol.y[1]**2)*0.5

if __name__ == '__main__':

    # Plotting p(t) and q(t)
    plt.figure(figsize=(10, 6))
    plt.plot(time, p_values, label='p(t)')
    plt.plot(time, q_values, label='q(t)')
    plt.xlabel('Time (t)')
    plt.ylabel('Function values')
    plt.title('Solution of the differential equations (Symplectic method)')
    plt.legend()
    plt.grid(True)
    plt.show()
