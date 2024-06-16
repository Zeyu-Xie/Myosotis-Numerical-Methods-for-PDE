import numpy as np
import matplotlib.pyplot as plt

# 参数设置
epsilon = 0.01
a = 0.5
b = 0.5
L = 2
T = 0.1
Nx = 50
Ny = 50
dt = 0.0001
dx = L / (Nx - 1)
dy = L / (Ny - 1)
Nt = int(T / dt)

# 初始化网格
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
u = np.zeros((Nx, Ny))

# 初始条件
for i in range(Nx):
    for j in range(Ny):
        if (x[i]**2 / a**2 + y[j]**2 / b**2 <= 1):
            u[i, j] = 1
        else:
            u[i, j] = -1

# 边界条件
u[:, 0] = u[:, -1] = u[0, :] = u[-1, :] = -1

# 迭代求解
for n in range(Nt):
    u_new = np.copy(u)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u_xx = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            u_yy = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            u_new[i, j] = u[i, j] + dt * (u_xx + u_yy + (1 / epsilon) * u[i, j] * (1 - u[i, j]**2))
    u = u_new

# 绘制结果
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u.T, levels=50, cmap='RdBu')
plt.colorbar()
plt.title(f'Reaction-Diffusion Solution with ε={epsilon}, a={a}, b={b}')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('result.png')
