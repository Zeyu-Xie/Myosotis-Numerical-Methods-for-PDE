import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_pde(N, epsilon):
    h = 1.0 / (N - 1)
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # Right-hand side function f(x, y) = sin(pi * x) * sin(pi * y)
    f = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Initialize the solution array
    u = np.zeros((N, N))

    # Construct the coefficient matrix A and right-hand side vector b
    A = np.zeros((N*N, N*N))
    b = np.zeros(N*N)
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            idx = i * N + j
            A[idx, idx] = -30 * epsilon / (12 * h**2)
            if i > 1:
                A[idx, idx - N] = 16 * epsilon / (12 * h**2)
                A[idx, idx - 2 * N] = -epsilon / (12 * h**2)
            if i < N - 2:
                A[idx, idx + N] = 16 * epsilon / (12 * h**2)
                A[idx, idx + 2 * N] = -epsilon / (12 * h**2)
            if j > 1:
                A[idx, idx - 1] = 16 * epsilon / (12 * h**2)
                A[idx, idx - 2] = -epsilon / (12 * h**2)
            if j < N - 2:
                A[idx, idx + 1] = 16 * epsilon / (12 * h**2)
                A[idx, idx + 2] = -epsilon / (12 * h**2)
            
            A[idx, idx - 1] -= 1 / (2 * h)
            A[idx, idx + 1] += 1 / (2 * h)
            
            b[idx] = f[i, j]
    
    # Boundary conditions u = 0 on the edges
    for i in range(N):
        A[i, i] = 1
        b[i] = 0
        A[(N-1)*N + i, (N-1)*N + i] = 1
        b[(N-1)*N + i] = 0
        A[i * N, i * N] = 1
        b[i * N] = 0
        A[i * N + (N - 1), i * N + (N - 1)] = 1
        b[i * N + (N - 1)] = 0

    # Solve the linear system
    u_flat = spsolve(A, b)
    u = u_flat.reshape((N, N))

    return X, Y, u, f

def compute_error(u, u_exact):
    return np.sqrt(np.sum((u - u_exact)**2) / np.size(u))

# Parameters
N = 50
epsilon_values = [1e-2, 1e-4, 1e-6]

for epsilon in epsilon_values:
    X, Y, u, f = solve_pde(N, epsilon)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.contourf(X, Y, u, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f'Solution with $\epsilon = {epsilon}$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(132)
    plt.contourf(X, Y, f, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Right-hand side $f(x,y)$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(133)
    plt.contourf(X, Y, np.abs(u - np.sin(np.pi * X) * np.sin(np.pi * Y)), levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Error $|u - f|$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(f'answer_{epsilon}.png')

    # Compute and print the error
    exact_solution = np.sin(np.pi * X) * np.sin(np.pi * Y)
    error = compute_error(u, exact_solution)
    print(f'Error for epsilon = {epsilon}: {error:.2e}')
