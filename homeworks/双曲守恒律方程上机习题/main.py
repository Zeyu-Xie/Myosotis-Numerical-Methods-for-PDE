import numpy as np
import matplotlib.pyplot as plt

# 定义初始条件
def initial_condition(x):
    return np.where((x >= 0.4) & (x <= 0.6), 1, 0)

# 定义 Godunov 格式
def godunov_flux(u_left, u_right):
    if u_left > u_right:
        if u_left <= 0:
            return 0.5 * u_left**2
        elif u_right >= 0:
            return 0.5 * u_right**2
        else:
            return 0
    else:
        if u_left >= 0:
            return 0.5 * u_left**2
        elif u_right <= 0:
            return 0.5 * u_right**2
        else:
            return 0

def godunov_step(u, dx, dt):
    flux = np.zeros(len(u) + 1)
    for i in range(1, len(u)):
        flux[i] = godunov_flux(u[i-1], u[i])
    return u - dt/dx * (flux[1:] - flux[:-1])

# 定义 Lax-Wendroff 格式
def lax_wendroff_step(u, dx, dt):
    f = 0.5 * u**2
    u_next = np.zeros_like(u)
    for i in range(1, len(u)-1):
        u_next[i] = u[i] - 0.5 * dt/dx * (f[i+1] - f[i-1]) + 0.5 * (dt/dx)**2 * (
            (u[i+1] + u[i]) * (f[i+1] - f[i]) - (u[i] + u[i-1]) * (f[i] - f[i-1])
        )
    return u_next

# 定义 TVD 格式
def minmod(a, b):
    if a * b <= 0:
        return 0
    else:
        return min(abs(a), abs(b)) * np.sign(a)

def tvd_step(u, dx, dt):
    u_next = np.zeros_like(u)
    flux = np.zeros_like(u)
    
    # 计算斜率
    slope = np.zeros_like(u)
    for i in range(1, len(u)-1):
        slope[i] = minmod((u[i] - u[i-1]) / dx, (u[i+1] - u[i]) / dx)
    
    # 计算界面值
    u_L = u - 0.5 * slope * dx
    u_R = u + 0.5 * slope * dx
    
    # 计算通量
    for i in range(1, len(u)):
        flux[i] = godunov_flux(u_L[i-1], u_R[i])
    
    u_next[1:-1] = u[1:-1] - dt/dx * (flux[1:-1] - flux[:-2])
    return u_next

# 设置网格和时间步长
x = np.linspace(0, 1, 201)
dx = x[1] - x[0]
dt = 0.5 * dx
t_final = 0.2

# 初始条件
u0 = initial_condition(x)

# Godunov 格式计算
u_godunov = u0.copy()
t = 0
while t < t_final:
    u_godunov = godunov_step(u_godunov, dx, dt)
    t += dt

# Lax-Wendroff 格式计算
u_lax_wendroff = u0.copy()
t = 0
while t < t_final:
    u_lax_wendroff = lax_wendroff_step(u_lax_wendroff, dx, dt)
    t += dt

# TVD 格式计算
u_tvd = u0.copy()
t = 0
while t < t_final:
    u_tvd = tvd_step(u_tvd, dx, dt)
    t += dt

# 绘图比较结果
plt.figure(figsize=(10, 6))
plt.plot(x, u0, label='Initial Condition', linestyle='--')
plt.plot(x, u_godunov, label='Godunov')
plt.plot(x, u_lax_wendroff, label='Lax-Wendroff')
plt.plot(x, u_tvd, label='TVD')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Comparison of Different Numerical Schemes')
plt.legend()
plt.grid()
plt.savefig('result.png')
