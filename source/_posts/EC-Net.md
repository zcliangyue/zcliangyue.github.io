---
title: EC-Net 论文阅读笔记
date: 2023-05-31 00:39:24
tags:
- 点云
- 点云上采样
- 神经网络
- 论文阅读笔记
mathjax: true
typora-root-url: ./EC-Net
index_img: /img/EC-Net.png
---

论文：[EC-Net: an Edge-aware Point set Consolidation Network](https://arxiv.org/pdf/1807.06010.pdf)

# 一、摘要及引言

点云整合（point cloud consolidation）是将点云“按摩”到表面上的过程，去噪、补全、重采样等等都属于其一部分。目前数据驱动方法展现出了很好的性能，但它们忽略了 3D 对象的锐利特征。该论文提出了第一个用于边缘感知整合网络 EC-Net ，其在 PointNet 的启发下，通过将坐标转换为深层特征并进行特征扩展产生更多点。同时论文提出了一种仅适用于点云的 patch 提取方案，以便提取 patch 并在训练和测试阶段共同使用。

<!--more-->

此外，为了使得网络具有**边缘感知**能力，论文将边缘和网格三角形信息同 patch 相关联，通过回归**点到边缘的距离**以及点坐标来训练网络。更重要的是，论文通过一种新颖的 edge-ware 联合损失函数，可以有效地比较输出点和基准 3D 网格。该损失函数鼓励输出点靠近底层表面和边缘，并更均匀地分布在表面上。

# 二、方法

## 训练数据准备

论文从 ShapeNet 和其它在线存储库收集 3D 网格 ，包括简单的 3D 形状、机械零件和椅子等日常用品。并且采用了手动绘制折线的方式来标记网格上的锐利边缘，用于学习点到边缘的距离，如下图所示。

<img src="9914334e1a334b6aa2fb96ea8b3c64f7.png" alt="网格边缘标记" style="zoom:80%;display: block; margin-left: auto; margin-right: auto;" />

### 虚拟扫描

为了从 3D 网格获取点云，论文采用了虚拟扫描的方式，具体来说。首先将网格归一化到 $\left[-1,+1\right]^3$ 单位距离空间里，并在物体周围水平均匀的布置 30 个虚拟相机（视角为 50°），相机距离物体中心两个单位距离，同时随机地扰动相机位置（向上或向下）。之后，通过渲染出网格对象的深度图并为深度值和像素位置添加量化噪声，最后反向投影每个像素得到点云。

这种方式模拟了真实场景里的扫描仪，比一般的下采样方法更加贴合实际。其特点是存在线状的点分布（下图左），离相机越近的平面采样点越密集（下图右），同时具有随机误差。

<img src="2a1c179521514327bbf9b1b4f61402ab.png" alt="虚拟扫描" style="zoom:80%;display: block; margin-left: auto; margin-right: auto;" />

### 局部区域提取

提取 patch 的依据是测地线距离，因为采用欧氏距离可能导致相邻点位于薄表面的两侧。具体的提取步骤如下：

- <img src="f19afa3e9ca74fdc978749eeb712767f.png" alt="在这里插入图片描述" style="zoom:90%;float:right;padding-left:30pt" />首先将每个点视为节点创建邻接图，图中每个点与其 $k$ 近邻点（$k=10$）相连，如右图所示。边权重设置为两点之间的欧氏距离；
- 随机选择 $m=100$ 个点作为 patch 的中心点。对于每个选定的点，通过 Dijkstra 算法找到 $2048$ 个在**邻接图上**最近的点；
- 在 $2048$ 个点里随机选取 $\hat{N}=1024$ 个点，从而引入随机性；
- 将点坐标归一化到单位球内，坐标重心为原点；
- 对于训练所用的 patch ，论文还找到了其附近相关联的网格三角形以及标记的边缘线作为基准信息。

## 边缘感知的点集整合网络

### 特征嵌入及扩展

该组件首先使用 PointNet++ 将每个点周围的局部信息映射到 $D$ 维特征向量中（论文实验取 $D=256$ ）。考虑到 patch 边缘点的局部信息不准确，论文只保留了 $N = \frac{\hat{N}}{2}$ 个最接近 patch 中心的点对应的特征向量，因此特征嵌入组件的输入为 $N\times 3$ 的张量。

下图为特征嵌入部分的详细结构，其中在四个级别分别采用了 $0.1、0.2、0.4、0.6$ 的分组半径来提取局部特征，这四个层次对应的点样本数分别为 $N,\frac{N}{2},\frac{N}{4},\frac{N}{8}$ 。这里虽然写的是 conv 层，但其实就是 PointNet++ 中的 SA 层。其特征组合方式参考了 PU-Net 。

<img src="537dcf63ed6d4f3f9c8a8729657f540c.png" alt="在这里插入图片描述" style="zoom:70%;display: block; margin-left: auto; margin-right: auto;" />

特征扩展部分和 PU-Net 一致，只不过在 PU-Net 中的 $1\times1$ 卷积，在这里也统一写成了卷积层。特征扩展部分的输出为 $rN\times128$ 的张量。

<img src="image-20230526201933967.png" alt="image-20230526201933967" style="zoom:70%;display: block; margin-left: auto; margin-right: auto;" />

### 边缘距离回归

这一组件将回归每个点到边缘的距离，从而进行边缘点识别。回归距离是点到最近的标记边缘的距离。具体来说，首先通过 MLP 从扩展特征中提取距离特征 $f_{dist}$ ，然后再通过另一个 MLP 回归点到边缘的距离（但是既然点是无序的，训练样本中的点到边缘距离如何排列？或者随机排列？）。

<img src="image-20230526202725949.png" alt="image-20230526202725949" style="zoom:70%;display: block; margin-left: auto; margin-right: auto;" />

### 坐标回归

基于边缘距离回归组件中的 $f_{dist}$ 特征，利用两个 MLP 回归点坐标。在这里论文采用了只回归**残差坐标**的方式，也就是上采样点相对当前点的偏移值，这样预测值会比较小，有利于网络学习。另外，论文提到除了最后用于回归的 MLP 层，其余所有卷积和 MLP 层后面都接的是 RELU 激活函数。

### 边缘点识别

记 $d_i$ 为输出点 $x_i$ 到边缘的距离，则边缘点集为 $\mathcal{S}_{\Delta_d}=\{x_i \}_{d_i<\Delta_d}$ 。该组件在训练和测试阶段都会执行。

## 边缘感知的联合损失函数

<img src="d39191aed6af41fab68697431bc668b7.png" alt="EC-Net架构图" style="zoom:70%;display: block; margin-left: auto; margin-right: auto;" />

损失函数的设计主要基于以下目标：1）靠近底层物体表面；2）靠近标记边缘；3）均匀分布

### 表面损失（Surface Loss）

表面损失定义为从每个输出点 $x_i$ 到与 patch 关联的所有网格三角形 $T$ 的最短距离：
$$
d_T(x_i,T)=\min_{t\in T}d_t(x_i,t),
$$
为了计算 $d_t$ ，需要考虑其中情况，因为三角形 $t$ 上离 $x_i$ 最近的点可能位于三角形的顶点、边缘或面内。计算所有输出点的 $d_T$ ，相加得到表面损失：
$$
L_{surf}=\frac{1}{\widetilde{N}}\sum_{1\leq i \leq \widetilde{N}}d_T^2(x_i,T),
$$
其中 $\widetilde{N}=rN$ ，为每个 patch 包含的点数。

### 边缘损失（Edge Loss）

边缘损失鼓励输出点位于靠近边缘的位置。将与 patch 相关的一组带注释的边缘段记作 $E$ ，定义边缘损失为从每个边缘点（边缘检测得到）到所有 patch 中的边缘段的最短距离的最小值：
$$
d_E(x_i,E)=\min_{e\in E}d_e(x_i,e)，
$$
其中 $d_e(xi,e)$ 是边缘点 $x_i$ 到边段 $e\in E$ 上任意点的最短距离。将所有边缘点的 $dE$ 求和，得到边缘损失：
$$
L_{edge}=\frac{\sum_{x_i\in \mathcal{S}_{\Delta_d}}d^2_E(x_i,E)}{\left|\mathcal{S}_{\Delta_d} \right|},
$$
总的思路和表面损失一样，只是表面损失针对所有点，边缘损失只针对边缘点。

### 排斥损失（Repulsion Loss）

排斥损失鼓励输出点更均匀分布。对于输出点 $x_i,i=1,\dots,\widetilde{N}$ ，排斥损失定义为：
$$
L_{repl}=\frac{1}{\widetilde{N}\cdot K}\sum_{1\leq i\leq \widetilde{N}}\ \sum_{i'\in \mathcal{K}(i)}\eta(\left\|x_{i'}-x_i \right\|),
$$
其中 $\mathcal{K}(i)$ 是 $x_i$ 的 $K$ 最近邻域的索引集合（论文设 $K=4$ ），而 $\eta(r)=\max(0,h^2-r^2)$ 是惩罚函数，两点越近则惩罚项越大。当距离大于 $h$ 时，则不起作用。

### 边缘距离回归损失（Edge Distance Regression Loss）

边缘距离回归损失引导网络回归 $rN$ 个输出点到边缘的距离 $d$ 。由于并不是每个点都靠近标记的边缘，因此回归损失应在一定距离截断，以免损失过大：
$$
L_{regr} = \frac{1}{\widetilde{N}}\sum_{1\leq i \leq \widetilde{N}}\left[\mathcal{T}_b(d_E(x_i,E))-\mathcal{T}_b(d_i) \right]^2
$$

### 端到端训练

联合损失函数定义为：
$$
\mathcal{L}=L_{surf}+L_{repl}+\alpha L_{edge}+\beta L_{regr}
$$

## 实施细节

### 网络训练

在训练之前，将每个输入补丁归一化到 $[-1, 1]^3$。然后通过一系列运算符在网络中即时对每个 patch 做数据增强：1）随机旋转；2）在所有维度上随机平移 $-0.2$ 到 $0.2$ ；3）随机缩放 $0.8$ 到 $1.2$ ；4）添加高斯噪声，参数 $\sigma$ 设置为补丁边界框大小的 $0.5\%$；随机调整补丁中点的顺序。

### 网络推理

这里主要阐述如何通过训练好的网络以 patch-wise （在图像分割中指介于像素和图像级别的区域）方式处理点云。

首先在测试点云中提取点集作为质心，以便使用 2.1 中的过程提取点块。为了使 patch 更均匀地分布在点云上（点云总点数为 $N_{pt}$ ，patch 点数为 $N$ ），使用最远点采样法，在测试点云中随机找到 $M = \beta \frac{N_{pt}}{N} $ 个点，这意味着平均每个点被采样了 $\beta$ 次，也就是平均每个点出现在 $\beta$ 个不同的 patch 中。

提取补丁后，将它们输入网络并应用网络生成 3D 坐标和点到边缘的距离，同时进行边缘点识别。与训练阶段的边缘点识别不同，这里设置了一个较小的阈值 $d =0.05$ 。在训练中则使用较大的 $d$ ，这是因为训练是一个优化过程，网络需要通过更多的点来学习识别边缘点。

### 表面重建

首先为网络的输出点构建一个 $k$ 近邻图。对于边缘点过滤，通过 RANSAC 拟合线段实现；对于表面点过滤，通过**边缘停止**的方式在 $k$ 近邻图中找到一小组附近的点，然后使用 PCA 拟合平面来实现。其中边缘停止指的是在到达边缘点时停止广度优先搜索（breath-first growth），这避免了越过边缘将无关点纳入。

重复多次以上步骤，最后通过纳入一些间隙中的原始点来填充边缘点和表面点之间的微小间隔，并通过投掷飞镖法（dart throwing）来添加新点。Dart throwing 是一种随机采样方法，在已有点的基础上，若新点离已有点太近（飞到镖盘内），则重新选取，直至点数达到要求。

为了进一步重建表面，论文采用了 EAR 中的方法对点云进行下采样并计算法线，使用球旋转（ball pivoting）或筛选泊松表面重建（screened Poisson surface reconstruction）来重建表面，并使用双边法线滤波（bilateral normal filtering）清洗生成的网格。这里提到的几个概念似乎属于三维重建，暂且将其相关论文放在这里，以后如果想起来学一下：

1. EAR：Edge-aware point set resampling
2. ball pivoting：The ballpivoting algorithm for surface reconstruction
3. screened Poisson surface reconstruction：Screened poisson surface reconstruction.
4. bilateral normal filtering：Bilateral normal filtering for mesh denoising

# 三、总结

EC-Net 主要内容可以概括如下：

- 为了使网络具有边缘感知能力，通过手动标注的方式在网格上标记边缘；
- 沿用了 PU-Net 中的特征嵌入和特征扩展模块
- 网络同时输出点到边缘距离、边缘点、上采样点，并分别设计了边缘距离损失、边缘损失、排斥损失以及表面损失，得到联合损失函数，用于端到端训练；
- 采用回归残差坐标的方式来得到上采样点坐标

EC-Net 主要存在以下不足：

- 和 PU-Net 一样不具备补全能力；
- 对于严重欠采样的微小结构，网络难以重建其锐利边缘；
- patch 点数固定，导致随着点云密度变化，其大小发生显著改变，难以适应不同规模的结构
