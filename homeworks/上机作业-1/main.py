import numpy as np
import conserved_RK4
import conserved_symplectic
import matplotlib.pyplot as plt
import os

T = 10
dt = 0.01
steps = int(T / dt)

with open(os.path.join(os.path.dirname(__file__), "config.txt"), "r") as file:
    T = float(file.readline())
    dt = float(file.readline())
    steps = int(T / dt)

time = np.linspace(0, T, steps)

p_1 = conserved_RK4.p_values
q_1 = conserved_RK4.q_values
p_2 = conserved_symplectic.p_values
q_2 = conserved_symplectic.q_values

if __name__ == '__main__':
    print(f"Conserved quantity using RK4: {conserved_RK4.conserved_RK4}")
    print(f"Conserved quantity using symplectic method: {conserved_symplectic.conserved_symplectic}")

    # Plotting p(t) and q(t)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

    ax1.plot(time, p_1, label='p(t)')
    ax1.plot(time, q_1, label='q(t)')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Function values')
    ax1.set_title('Solution of the differential equations')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time, p_2, label='p(t)')
    ax2.plot(time, q_2, label='q(t)')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Function values')
    ax2.set_title('Solution of the differential equations (Symplectic method)')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(time, conserved_RK4.conserved_RK4, label='Conserved quantity (RK4)')
    ax3.plot(time, conserved_symplectic.conserved_symplectic, label='Conserved quantity (Symplectic)')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Conserved quantity')
    ax3.set_title('Conserved quantity (RK4)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "results.png"))
    plt.show()