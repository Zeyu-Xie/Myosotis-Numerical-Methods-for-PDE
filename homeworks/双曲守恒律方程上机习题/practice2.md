# 双曲守恒律问题上机习题

考虑以下 Burgers' 方程的初值问题
$$
\begin{equation}
\left\{
\begin{aligned}
\frac{\part u}{\part t}+\frac{\part u}{\part x}\big(\frac{u^2}{2}\big) &= 0 \\
u(x,0) &= u_0(x)
\end{aligned}
\right.
\end{equation}
$$
取以下初值
$$
u_0(x) = \begin{equation}
\left\{
\begin{aligned}
1 &\quad x\in[0.4, 0.6] \\
0 &\quad x\notin[0.4, 0.6]
\end{aligned}
\right.
\end{equation}
$$
进行计算

试分别用 Godunov 格式、Lax-Wendroff 格式和一种二阶 TVD 格式进行计算，并比较结果
