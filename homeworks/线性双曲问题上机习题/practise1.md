# 线性双曲问题上机习题

考虑以下对流问题
$$
\begin{equation}
\left\{
\begin{aligned}
\frac{\part u}{\part t}+\frac{\part u}{\part x} &= 0 \\
u(x,0) &= u_0(x)
\end{aligned}
\right.
\end{equation}
$$
分别取以下两种初值
$$
u_0(x) = \begin{equation}
\left\{
\begin{aligned}
1- cos2\pi x &\quad x\in[0, 1] \\
0 &\quad x\notin[0, 1]
\end{aligned}
\right.
\end{equation}
$$
和
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
进行计算。例如算到 $T=1$，可以取 $x\in[0, 2]$ 计算即可，对于 $x\notin[0,2]$，可以取 $u(t, x) = 0$

试分别用迎风格式、Lax-Friendrichs 格式、Lax-Wendroff 格式和 Beam-Warming 格式进行计算，比较并分析所得结果
