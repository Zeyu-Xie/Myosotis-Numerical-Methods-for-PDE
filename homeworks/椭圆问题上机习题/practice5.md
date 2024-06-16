# 椭圆问题上机习题

考虑以下奇异振动问题
$$
\begin{equation}
\left\{
\begin{aligned}
-\epsilon\Delta u(x,y) + \frac{\part u}{\part x}(x,y) &= sin(\pi x)sin(\pi y)\quad (x,y)\in\Omega \\
u|_{\part\Omega} = 0
\end{aligned}
\right.
\end{equation}
$$
其中 $0<\epsilon\ll 1$，且 $\Omega = [0, 1]\times[0, 1]$

试构造一种对于 $\epsilon\ll h$ 都能保持高精度的离散格式，可取不同的 $\epsilon$ 和步长 $h$ 计算，比较结果