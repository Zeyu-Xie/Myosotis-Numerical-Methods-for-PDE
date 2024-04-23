import numpy as np
import matplotlib.pyplot as plt
import os


def RK4_step(f, t, y, dt):
    k1 = dt * f(t, y)
    k2 = dt * f(t + 0.5*dt, y + 0.5*k1)
    k3 = dt * f(t + 0.5*dt, y + 0.5*k2)
    k4 = dt * f(t + dt, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


def f(t, y):
    p, q = y
    return np.array([-q, p])

T = 10
dt = 0.01
steps = int(T / dt)

with open(os.path.join(os.path.dirname(__file__), "config.txt"), "r") as file:
    T = float(file.readline())
    dt = float(file.readline())
    steps = int(T / dt)

# Initial conditions
y = np.array([0, 1])

# Storage for results
results_RK4 = np.zeros((steps, 3))

# Main loop for RK4
for i in range(steps):
    t = i * dt
    y = RK4_step(f, t, y, dt)
    results_RK4[i] = [t, y[0], y[1]]

# Extracting results
time = results_RK4[:, 0]
p_values = results_RK4[:, 1]
q_values = results_RK4[:, 2]

# Computing conserved quantity
conserved_RK4 = (results_RK4[:, 1]**2 + results_RK4[:, 2]**2)*0.5

if __name__ == '__main__':

    # Plotting p(t) and q(t)
    plt.figure(figsize=(10, 6))
    plt.plot(time, p_values, label='p(t)')
    plt.plot(time, q_values, label='q(t)')
    plt.xlabel('Time (t)')
    plt.ylabel('Function values')
    plt.title('Solution of the differential equations')
    plt.legend()
    plt.grid(True)
    plt.show()
