---
title: DDIM简介
date: 2025-03-30 00:20:00
tags:
- Diffusion
- 神经网络

math: MathJax
mathjax: true
index_img: /img/DDIM.png
---

<!--more-->

# 引子

当思维被定式成DDPM那一套之后，DDIM可能不是一个太容易理解的东西。首先我们用最快的方式回顾DDPM的推导，从贝叶斯定理开始，我们希望用前向分布估计反向分布：
$$
p(x_{t-1}|x_t)=\frac{p(x_{t}|x_{t-1})p(x_{t-1})}{p(x_t)},
$$
由于 $p(x_{t-1}),p(x_t)$ 的不确定，退而求其次以 $x_0$ 作为额外的条件，得到：
$$
p(x_{t-1}|x_t,x_0) =\frac{p(x_{t}|x_{t-1},x_0)p(x_{t-1}|x_0)}{p(x_t|x_0)},
$$
这样一来，等式右边所有的分布都可以被写作高斯分布，相乘或相除之后依然是高斯分布。而对于等式左边，还差一个 $x_0$ 不知道。因此可以将 $x_0$ 参数化为 $x_t$ 和噪声 $\epsilon$ 的组合，由神经网络负责预测 $\epsilon$ ，从而实现 $x_{t}$ 推算 $x_{t-1}$ ，即反向去噪。

上述过程看起来可以说行云流水。但一个潜在的问题是：既然神经网络预测了噪声 $\epsilon$ ，为什么不能直接返回 $x_0$ 或者 $x_s(s<t)$ 呢？他们也是 $x_t$ 与 $\epsilon$ 的组合，如果可以“跳步”，岂不是可以更快速地采样？这个问题的另一个表述是：DDPM为什么一定要一步一步去噪？

我们不妨来试一试。假设 $s<t$，且 $t-s>1$ ，那么可以得到：
$$
p(x_s|x_t,x_0)=\frac{p(x_{t}|x_s,x_0)p(x_s|x_0)}{p(x_t|x_0)},
$$
右侧的三个分布依然是高斯分布。这表明 DDPM 是可以跳步采样的。但由于 DDPM 路径的随机性，跳步采样通常意味着误差的累积，导致生成质量快速下降。

而DDIM提供了一个解决办法：直接假设目标分布 $p(x_s|x_t,x_0)$ 为高斯分布，放弃前向过程的马尔科夫性质。这样一来，相当于不知道 $p(x_t|x_{t-1})$ 的分布。让我们看看DDIM是如何导出一个新的采样公式的。

# DDIM的待定系数法

既然我们假设了高斯分布，那么仿照DDPM的推导结果，可以设置三个系数，将其表示为：
$$
p(x_s|x_t,x_0) = \mathcal{N}(x_s;kx_0+mx_t, \sigma^2\mathbf{I}),
$$
根据重参数化，可以将 $x_s$ 写成 $kx_0+mx_t+\sigma\epsilon$ ，另一方面 $x_s = \sqrt{\overline{\alpha}_s}x_0 + \sqrt{1-\overline{\alpha}_s}\epsilon$ 。由于 DDIM 的目标是保持前向过程每个阶段的 $x_t$ 分布不变，我们可以从均值和方差两个方面构建这两者之间的等价性。

首先，$x_t = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\epsilon$ ，因此
$$
x_s = (k+m\sqrt{\overline{\alpha}_t})x_0 + \sigma\epsilon_1+m\sqrt{1-\overline{\alpha}_t}\epsilon_2,
$$
服从均值为 $(k+m\sqrt{\overline{\alpha}_t})x_0$ ，方差为 $\sqrt{\sigma^2+m^2(1-\overline{\alpha}_t)}$ 的高斯分布。因此有对应的等式：
$$
\begin{aligned}
k+m\sqrt{\overline{\alpha}_t} = \overline{\alpha}_s \\
\sigma^2+m^2(1-\overline{\alpha}_t) = 1 - \overline{\alpha}_s
\end{aligned}
$$
两个方程，三个未知数，DDIM将 $\sigma$ 作为可变参数，解出 $k$ 和 $m$：
$$
m=\frac{\sqrt{1-\overline{\alpha}_s-\sigma^2}}{\sqrt{1 - \overline{\alpha}_t}},\qquad k=\overline{\alpha}_s-\frac{\sqrt{1-\overline{\alpha}_s-\sigma^2}}{\sqrt{1 - \overline{\alpha}_t}}\overline{\alpha}_t.
$$
这样，我们就导出了采样公式：
$$
x_s = \sqrt{\overline{\alpha}_s}\left(\frac{x_t-\sqrt{1-\overline{\alpha}_t}\epsilon_{\theta}(x_t, t)}{\sqrt{\overline{\alpha}_t}}\right)+\sqrt{1-\overline{\alpha}_s-\sigma^2}\epsilon_{\theta}(x_t, t)+\sigma\epsilon.\tag{1}\label{1}
$$
这个采样公式也可以退化成 $s=t-1$ 的形式。

可以看到，神经网络 $\epsilon_\theta$ 的定义和DDPM完全一致。事实上，DDIM只是一种采样方法，并不影响训练的目标。换言之，对于任意训练好的diffusion模型，只需要换一个采样公式，就可以实现跳步的采样，因为脱离了马尔科夫的要求。

此外，也不必担心DDIM不匹配前向过程，无论 $\sigma$ 取何值，前向分布 $p(x_t|x_0)$ 还是原定的高斯分布，由 $\alpha_t$ 给出。

## 不同的$\sigma$

回顾DDPM的采样公式：
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\overline{\alpha}_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_{\theta}(x_t, t)\right)+\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_t\epsilon.
$$
令DDIM中的 $\sigma=\frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_t$ ，则可以得到和上式一样的结果，这严格表明了DDIM和DDPM并不冲突，是一种更灵活的形式。

更特别地，令 $\sigma=0$，则反向采样过程变成确定性的，采样的 $x_T$ 确定时，$x_0$ 随即确定。在确定性采样时，能够取得最好的加速结果，也就是采样步数对生成质量的影响最小。直观上来说，确定性的轨迹会更简单，无需那么多的步骤去拟合。但另一方面，模型终究只是分布的近似，因此即便是确定性的，依然是步数越多质量越好。

而在DDPM中，方差的选取是被回避的，直接通过计算得到，这是因为方差必须是这个值，如果变了就和前向过程相互矛盾。因此一句话总结：**DDIM让方差变成一个可变值，而为了使其和前向过程依然保持一致，需要改变前面的系数。**

# DDIM Inversion

Inversion（反转）是一个很有趣的话题。当我们能够从噪声中采样得到图像，人们开始思考另一个问题：如果我只有一张干净的图像，以及一个扩散模型，能够对图像加入合适的噪声，并通过模型恢复出原始图像。

这个需求听起来很奇怪，但它在图像编辑/转换之类的领域中非常重要。往图像中添加噪声，再通过合适的引导重新恢复，能够实现对图像的编辑。并且这类方法通常只需要涉及对扩散路径的准确恢复，无需任何额外的训练。

换言之，给定 $x_0$ ，inversion 的目的是找出噪声路径 $x_0\rightarrow x_1 \rightarrow\cdots\rightarrow x_T$ ，然后基于去噪器和 DDIM 的确定性采样（$\sigma=0$）来重新恢复 $x_0$。让我们将采样公式 $\eqref{1}$ 改写一下，将 $x_t$ 变成关于  $x_s$ 的函数：
$$
x_t = \sqrt{\overline{\alpha}_t}\left(\frac{x_s-\sqrt{1-\overline{\alpha}_s}\epsilon_{\theta}(x_t, t)}{\sqrt{\overline{\alpha}_s}}\right)+\sqrt{1-\overline{\alpha}_t}\epsilon_{\theta}(x_t, t).
$$
这里还差的一步是 $\epsilon_{\theta}(x_t,t)$ 实际上是未知的，因为我们有的仅仅是更干净的 $x_s$ 。最简便的方法是假设 $\epsilon_{\theta}(x_t,t)\approx \epsilon_{\theta}(x_s,t)$ ，这样一来等式右边就都不包含未知的 $x_t$ 了。步数越多，$s$ 和 $t$ 越接近，inversion的准确性自然也就越高，恢复出的图像（用算出来的带噪声图像重新去噪得到）与原图像就越像。

受益于 DDIM 的确定性采样，用来做 Inversion 可以说再合适不过。相较而言，用 DDPM 做 Inversion 就没有那么容易。同理，SDE 和 Flow matching 也有相应的 inversion 方法，此处不做深入展开。













