---
title: SDE：重新理解DDPM和SMLD
date: 2025-03-25 00:44:00
tags:
- Diffusion
- 神经网络

math: MathJax
mathjax: true
index_img: /img/SDE_ODE.png

---

参考资料：

- [生成扩散模型漫谈（五）：一般框架之SDE篇](https://spaces.ac.cn/archives/9209)
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103)

<!--more-->

# 引子

通常来说，初次接触DDPM是一个让人困惑的事情，因为似乎有许多不太一样的扩散模型，他们和DDPM的区别有些难以捉摸。这对于一个想要构建起知识体系的学习者来说是糟糕的。因此，通过SDE阐述扩散模型的方法非常值得学习，它提供了一个更一般的视角，使我们不必再依赖各种各样的直觉。

让我们先从扩散模型的前向过程开始：
$$
x_t = \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t,
$$
这是一个迭代方程，每一个新的状态都是基于上一个状态进行更新的。这类型（即迭代）的方程通常可以和**常微分方程**（ODE）——即未知函数只包含一个自变量的微分方程——联系起来。例如，考虑一个最简单的匀速直线运动，每隔时间 $\Delta t$ 更新一次位移，则
$$
S_{t+1}=S_t + \Delta S,
$$
其中 $\Delta S$ 是一个常数。这个方程也可以写成连续的形式：
$$
\frac{S(t+\Delta t) - S(t)}{\Delta t} = \frac{\Delta S}{\Delta t}=v,
$$
当 $\Delta t\rightarrow 0$ 时，方程写作 $\mathrm{d}S/\mathrm{d}t=v$ 。虽然这个例子有些过于简单，但是足够了。

同理，既然扩散过程也是迭代，是否可以被写成微分方程？由于扩散过程具有随机性，需要借助**随机微分方程**（SDE）。

# 随机微分方程

为了假装比较专业，我们先从一些标准形式开始。首先是常微分方程：
$$
\mathrm{d}x = f(t, x)\mathrm{d}t
$$
其中 $x$ 是关于 $t$ 的函数。常微分方程表明函数的导数不仅与自变量 $t$ 有关，还和当前函数值 $x(t)$ 有关。仅此而已。

这个函数是确定性的，如果再加一个随机项：
$$
\mathrm{d}x = f(t, x)\mathrm{d}t + g(t, x) \mathrm{d}w,\tag{1}\label{1}
$$
这就得到了随机微分方程，其中 $\xi(t)$ 是一个噪声函数。此外，这里的 $w$ 是**布朗运动**，它满足以下性质：

- 初始条件：$w(0)=0$
- 独立增量：对于任意 $0\le t_1 < t_2 < \cdots < t_n$ ，布朗运动的增量 $w(t_2-t_1), w(t_3-t_2),\cdots,w(t_n-t_{n-1})$ 是相互独立的。
- 增量正态分布： 对于任意 $t$ 和 $\Delta t>0$ ，布朗运动的增量服从均值为 $0$ ，方差为 $\Delta t$ 的正态分布：

$$
w(t+\Delta t) - w(t) \sim \mathcal{N}(0, \Delta t).
$$

- 连续性：布朗运动是连续的，没有跳跃。

根据增量正态分布的性质，可以得到 $\mathrm{d}w\sim\mathcal{N}(0, \mathrm{d}t)$ ，理论上与 $\xi(t)\sqrt{\mathrm{d}t}$ 同分布。这就是为什么将离散SDE写作
$$
x_{t+\Delta t} - x_t = f(t, x_t)\Delta t + g(t, x_t) \sqrt{\Delta t} \epsilon, \qquad \epsilon\sim \mathcal{N}(0, \mathbf{I}).
$$
这里我们不推导其与微分形式$\eqref{1}$的等价性，因为常规的微分方法其实不适用于布朗运动（微分的方差无穷大），这一块我就完全不懂了。

# 前向过程

下面我们尝试将前向过程写成SDE的形式。为了将其连续化，首先定义一个连续的噪声表，步长 $\Delta t= 1/N$ ，因此 $\beta(\frac{t}{N})=\beta_t$ 。此外，为了统一自变量的范围，规定 $\beta(t/N)= \beta(t)/N = \beta(t)\Delta t$ 。

从常规的迭代公式开始，
$$
x_t = \sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_{t-1},
$$
我们代入连续的噪声表：
$$
x_t = \sqrt{1-\beta\left(\frac{t}{N}\right)}x_{t-1} + \sqrt{\beta\left(\frac{t}{N}\right)} \epsilon_{t-1},
$$
同样，其它变量也变成连续形式：
$$
x(t+\Delta t) = \sqrt{1-\beta(t)\Delta t}x(t) + \sqrt{\beta(t)\Delta t} \epsilon(t),
$$
下面做一些近似：
$$
x(t+\Delta t) \approx \left(1-\frac{1}{2}\beta(t)\Delta t\right)x(t) + \sqrt{\beta(t)\Delta t} \epsilon(t),
$$
整理得到：
$$
x(t+\Delta t)-x(t) \approx -\frac{1}{2}\beta(t)\Delta tx(t) + \sqrt{\beta(t)}\sqrt{ \Delta t}\epsilon(t),
$$
这就得到了离散SDE形式。我们将其转换为连续的：
$$
\mathrm{d}x = -\frac{1}{2}\beta(t)x\Delta t + \sqrt{\beta(t)}\mathrm{d}w,\tag{2}\label{2}
$$

即 $f(x,t)=-\frac{1}{2}\beta(t)x, \quad g(x, t)= \sqrt{\beta(t)}$ 。

# 逆向随机微分方程

任何SDE都有一个相应的逆向SDE。了解这一点其实也就足够了，可以直接跳到结果部分。下面给出一个逆向过程的推导，参考[生成扩散模型漫谈（五）：一般框架之SDE篇](https://spaces.ac.cn/archives/9209)。

对于一般形式$\eqref{1}$的SDE，我们可以将其写成Diffusion中条件概率的形式：
$$
p(x_{t+\Delta t}|x_t) = \mathcal{N}\left(x_t+f(t, x)\Delta t, g^2(t, x)\Delta t\right)
$$

为了寻求逆分布，我们也采用DDPM中相同的推导方式，即贝叶斯定理：
$$
p(x_t|x_{t+\Delta t}) = \frac{p(x_{t+\Delta t}|x_t)p(x_t)}{p(x_{t+\Delta t})}=p(x_{t+\Delta t}|x_t)\exp\left(\log p(x_t) - \log p(x_{t+\Delta t})\right)
$$
代入高斯分布得：
$$
p(x_t|x_{t+\Delta t}) \propto \exp\left(-\frac{\|x_{t+\Delta t}-x_t-f(x, t)\Delta t\|^2}{2g^2(x,t)\Delta t} + \log p(x_t) - \log p(x_{t+\Delta t})\right)
$$
由于我们关心的是 $\Delta t\rightarrow 0$ 的情形，可以将 $\log p(x_{t+\Delta t})$ 做泰勒展开：
$$
\log p(x_{t+\Delta t})\approx \log p(x_t) + (x_{t+\Delta t} - x_t)\cdot \nabla_{x_t}\log p(x_t) + \Delta t \frac{\partial}{\partial t}\log p(x_t)
$$
代入得：
$$
p(x_t|x_{t+\Delta t}) \propto \exp\left(-\frac{\|x_{t+\Delta t}-x_t-f(x, t)\Delta t\|^2}{2g^2(x,t)\Delta t} - (x_{t+\Delta t} - x_t)\cdot \nabla_{x_t}\log p(x_t) - \Delta t \frac{\partial}{\partial t}\log p(x_t)\right)
$$
将 $(x_{t+\Delta t} - x_t)\cdot \nabla_{x_t}\log p(x_t)$ 合并到前面的分子中，并省去 $\Delta t$ 的二次项，可得
$$
\begin{aligned} 
p(x_t|x_{t+\Delta t}) \propto&\, \exp\left(-\frac{\Vert x_{t+\Delta t} - x_t - \left[f(x_t, t) - g(x_t, t)^2\nabla_{x_t}\log p(x_t) \right]\Delta t\Vert^2}{2 g(x_t, t)^2\Delta t}\right) \\ 
\approx&\,\exp\left(-\frac{\Vert x_t - x_{t+\Delta t} + \left[f_{t+\Delta t}(x_{t+\Delta t}) - g_{t+\Delta t}^2\nabla_{x_{t+\Delta t}}\log p(x_{t+\Delta t}) \right]\Delta t\Vert^2}{2 g_{t+\Delta t}^2\Delta t}\right) 
\end{aligned}
$$
换回到SDE的形式，可得
$$
x_t = x_{t+\Delta t} - \left[f_{t+\Delta t}(x_{t+\Delta t}) - g^2(x_{t+\Delta t}, t+\Delta t)\nabla_{x_{t+\Delta t}}\log p(x_{t+\Delta t}) \right]\Delta t - g(x_{t+\Delta t}, t+\Delta t)\sqrt{\Delta t}\epsilon.
$$
注意等式右边不包含 $x_t$，因为我们必须用 $x_{t+\Delta t}$ 来导出 $x_t$ 。对于微分形式，则全部都用 $x$ 即可：
$$
\mathrm{d}x = \left[f(x, t)-g^2(x, t)\nabla_{x}\log p(x)\right]\mathrm{d}t + g(x, t)\mathrm{d}w.\tag{3}\label{3}
$$
注意这里等式右边的符号。反向过程和 $\mathrm{d}x$ 的方向是相反的，所以右边又反转了一次正负号。

# 反向去噪

结合前向过程的SDE形式$\eqref{2}$以及标准的反向SDE方程$\eqref{3}$ ，可以给出DDPM反向去噪的SDE方程：
$$
\mathrm{d}x = -\beta(t)\left[\frac{x}{2}+\nabla_{x}\log p(x)\right]\mathrm{d}t + \sqrt{\beta(t)}\mathrm{d}w.
$$
实际应用时，回到离散的形式：
$$
\begin{equation}
x_{t} - x_{t+\Delta t} =-\beta(t+\Delta t)\left[\frac{x_{t+\Delta t}}{2}+\nabla_{x_{t+\Delta t}}\log p(x_{t+\Delta t})\right]\Delta t + \sqrt{\beta(t+\Delta t)}\sqrt{\Delta t}\epsilon_t.\tag{4}\label{4}
\end{equation}
$$
这个式子中未知的部分是 $\nabla_{x_{t+\Delta t}}\log p(x_{t+\Delta t})$ ，也就是SMLD中的**分数**。这里做一个速通：

- SMLD的核心思想是通过预测对数似然函数的梯度 $\nabla_x\log p(x)$，然后利用朗之万方程（带有随机项的梯度下降）从分布中采样。
- 为了实现对分数的预测，需要用到分数匹配的技术。其中最流行的是去噪分数匹配，通过构建条件分布的分数来监督神经网络。
- 在推理过程中，采用退火采样，在初期用较大的噪声采样，后期用较小的噪声，避免在初期陷入低密度区域（预测不准），导致错误的结果。

如果我们沿着$\eqref{4}$推导出离散的反向过程，会发现它和DDPM完全一致，这当然符合我们的预期：从前向过程推导出$f$ 和 $g$ ，然后应用到逆向SDE的公式中，应当符合反向过程的SDE。这被称为**方差保留**（VP）SDE。

另一方面，SMLD中的反向过程并不是这么做的，说明其前向应当也是不一样的。在SMLD中并没有真正意义上定义过前向过程，因为它是直接从高斯分布出发的。但注意到它在训练时采用了一系列噪声尺度 $\sigma_i(i=1,2,\dots,N)$ ，我们可以手动给出一个马尔科夫链：
$$
x_t = x_{t-1} + \sqrt{\sigma^2_t-\sigma^2_{t-1}}\epsilon_{t-1}.
$$
这使得当 $x_{t-1}$ 的方差为 $\sigma_{t-1}$ 时，$x_t$ 的方差为 $\sigma_t$ 。我们同样可以根据这个式子，导出其SDE形式。根据
$$
x(t+\Delta t) = x(t) + \sqrt{\sigma(t+\Delta t)^2 - \sigma(t)^2}\epsilon_t，
$$
当 $\Delta t\rightarrow 0$ 时，有
$$
x(t+\Delta t) = x(t) +  \sqrt{\frac{\mathrm{d[\sigma(t)^2]}}{\mathrm{d}t}\Delta t} \epsilon_t，
$$
因此 $f(x, t)=0, g(x,t)= \sqrt{\frac{\mathrm{d[\sigma(t)^2]}}{\mathrm{d}t}}$ ，则SDE为：
$$
\mathrm{d}x = \sqrt{\frac{\mathrm{d[\sigma(t)^2]}}{\mathrm{d}t}}\mathrm{d}w.
$$
紧接着，直接导出反向SDE：
$$
\mathrm{d}x = -\left[\frac{\mathrm{d[\sigma(t)^2]}}{\mathrm{d}t}\nabla_{x}\log p(x)\right]\mathrm{d}t +\sqrt{\frac{\mathrm{d[\sigma(t)^2]}}{\mathrm{d}t}}\mathrm{d}w.
$$
将 $\frac{\mathrm{d[\sigma(t)^2]}}{\mathrm{d}t}$ 简记为 $\alpha(t)$ ，则有
$$
\mathrm{d}x = -\alpha(t)\nabla_{x}\log p(x)\mathrm{d}t +\sqrt{\alpha(t)}\mathrm{d}w.
$$
我们就得到了**朗之万方程**，由此验证SMLD也可以表达为SDE以及逆向SDE。对应的，这被称作**方差爆炸**（VE）SDE。



