# 抛物型方程的差分方法实验题

考虑以下的反应扩散问题：
$$
\begin{equation}
\left\{
\begin{aligned}
\frac{\part u}{\part t} &= \frac{\part^2 u}{\part x^2} + \frac{\part^2 u}{\part y^2}+\frac{1}{\epsilon}u(1-u^2)\quad(x,y)\in\Omega, t>0 \\
u|_{\part\Omega} &= -1 \\
u|_{t=0} &= \left\{
\begin{aligned}
1 &\quad x\in\widetilde{\Omega} \\
-1 &\quad x\in\Omega - \widetilde{\Omega}
\end{aligned}\right.
\end{aligned}
\right.
\end{equation}
$$
其中 $0<\epsilon\ll 1$，$\Omega = [-1, 1]\times[-1, 1]$，$\widetilde{\Omega} = \left\{(x, y)\in\mathbb{R}^2\vert\frac{x^2}{a^2}+\frac{y^2}{b^2} \leq 1\right\}$

且 $0<a<1$，$0<b<1$

试构造一种无条件稳定的（当然稳定性也不依赖于 $\epsilon$）二阶格式，并取不同的 $\epsilon, a, b$ 和充分小的步长 $h$ 计算，看看结果如何有什么不同

提示：常微分方程
$$
\frac{du}{dt} = \frac{1}{\epsilon}u(1-u^2)
$$
可以得到精确解的解析表达式