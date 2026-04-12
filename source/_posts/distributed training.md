---
title: Slurm 分布式训练
date: 2025-6-6 14:00
tags:
- 分布式训练
- 神经网络
index_img: /img/distributed.PNG
math: MathJax
mathjax: true
typora-root-url: ./distributed training

---

在公司实习的第一件任务就是部署多机多卡训练。在原始代码已经提供了对分布式训练的支持的基础上，这篇笔记主要梳理了如何通过 slurm 从后台提交多机多卡训练任务。

<!--more-->

# 分布式训练简介

<img src="distri.PNG" style="zoom:50%; display: block; margin-left: auto; margin-right: auto;" />

对于分布式训练的简单概念，可以直接参考 ColossalAI 的[文档](https://ox1xq40d2or.feishu.cn/docx/Iy82dJqN5opAzPxy9x3cO0x5nwh#share-RNJXdu5spogU6pxGz80ceog5n4g) 。简单来说，由于单张 GPU 的显存不够，无法训练较大的模型，并且由于 batch size 小，训练效率也大打折扣。通过将多个 GPU 连接起来，我们可以利用不同的分布式训练技巧实现：

- 每张 GPU 上更小的显存占用；
- 总体更大的吞吐量。

简单来说，可以将并行方法分为数据并行和模型并行两类。其中数据并行将模型加载到所有 GPU 上，然后将输入的 batch 分给不同 GPU，分别进行梯度计算后汇总更新。数据并行能够实现更大的 batch size，但很难节省每张 GPU 的显存。因此当模型大到一定程度的时候，无法仅依赖数据并行。而模型并行则涉及到更多复杂的方法，它可以拆分张量、模型组件和 token 序列等等，可以缓解单张 GPU 显存不足的问题。

# 集群多机训练部署

公司使用 SLURM 系统管理集群，SLURM 主要用于调度集群中的计算资源，包括 CPU、GPU 和内存等等。SLURM 系统的好处是任务提交后就不用管了，不需要 tmux 之类的来保留终端。所有的信息会输出在一个后缀为 `.out` 的日志文件中。另一方面，其他使用集群的人也可以查看到所有正在进行的任务（`squeue`）、节点的状态（`sinfo -N`）等等。

SLURM 另一个好处是，他可以指定节点去提交任务。在运行多机任务的时候，需要同时向多台节点提交任务，如果一个一个去提交，显然很麻烦。

## 训练脚本

首先，我们需要一个支持多机多卡并行的代码。MagicDriveDiT 使用 ColossalAI 提供的接口来实现数据并行和序列并行（Sequence Parallel）。序列并行是模型并行的一种，可以将一个 batch 进一步划分，节省每张 GPU 的显存占用，但会导致速度变慢将近一倍。只有在显存确实不足的时候使用。

`SLURM` 脚本主要可以分为几个部分。在文件头部，声明 `SLURM` 系统相关的环境变量，例如 CPU 线程数，每个节点的任务数，每个节点的 GPU 数量，节点名称等等。

### 资源分配

```bash
#!/bin/bash  # 表示将由解释器 /bin/bash 来执行该脚本
#SBATCH --account=huawei # 指定作业所属的用户（默认就是提交时候的用户）
#SBATCH --cpus-per-task=64 # 每个任务分配的 CPU 核心数量
#SBATCH --gres=gpu:8 # 每个节点请求 8 个 GPU。gres 表示通用资源（Generic RESources）
#SBATCH --job-name=magicdrive # 任务名称
#SBATCH --nodelist=node110,node111 # 指定作业运行节点
#SBATCH --nodes=2 # 请求两个节点
#SBATCH --ntasks-per-node=1 # 每个节点只运行 1 个任务（通常用于多机分布式训练）
#SBATCH --open-mode=append # SLURM 输出文件是追加模式写入（默认是覆盖）。
#SBATCH --partition=gpu_A6000 # 选择名为 gpu_A6000 的分区，表示运行在集群中搭载 gpu_A6000 的节点
#SBATCH --signal=USR2@120 # 用于在任务结束前 120 返回监听信号，可以触发某个函数
#SBATCH --time=20160 # 最大运行时间 20160 分钟
```

这些配置用于 SLURM 变量，主要的功能是分配计算资源。

### 环境配置

在此之后，我们需要配置系统环境变量，例如 `CUDA`、`CONDA`等。这里需要注意，`PATH` 使用追加的方式添加路径，而 `LD_LIBRARY_PATH` 和 `CUDA_HOME` 需要采用覆盖的方式，否则一些库在调用 `CUDA` 路径的时候会出现错误。

```bash
# CUDA=11.8
export PATH=/starmap/nas/cuda/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/starmap/nas/cuda/cuda-11.8/lib64  
export CUDA_HOME=/starmap/nas/cuda/cuda-11.8
```

然后设置 conda 虚拟环境。虽然在 master 终端提交任务时，会自动在所有节点上激活同样的环境。但设置一个 conda 激活的命令，就不需要在 master 节点上打开环境了。

```bash
# conda env
source /starmap/nas/anaconda3/etc/profile.d/conda.sh
conda activate magicdrivedit
```

然后加上一些可能需要的环境变量，用于定义线程数

```bash
export OMP_NUM_THREADS=8
export SUBMITIT_EXECUTOR=slurm # 用 submitit 管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 减少显存碎片
```

这里的 `OMP_NUM_THREADS` 表示程序在使用 `OpenMP` 并行计算时，每个进程使用的线程数，这个参数很关键。`OpenMP` 用于在多核 CPU 上并行加速计算任务，比如矩阵乘法、张量操作等。许多底层科学计算库（如 `NumPy`、`MKL`、`OpenBLAS`、`PyTorch`、`TensorFlow`）内部都用到了 `OpenMP` 或类似技术进行加速。

关于如何设置该参数：首先，我们在每个节点上启用了 8 张 GPU，64 个 CPU 线程。而通常来说，一个 GPU 对应于一个进程，这样更容易推广到多机多卡通讯。这个会在后面通过 `torchrun` 的 `--nproc_per_node=8` 进行设置。所以每个进程可以分配到 8 个线程，那么应当设置 `OMP_NUM_THREADS` 为 8，以最大化利用其并行计算能力。

### NCCL 配置

多机通讯的方法很多，主要通过各种网络协议实现。我们的集群采用的是 InfiniBand。InfiniBand (IB) 是一种高性能的计算机网络通信标准，主要用于高吞吐量和低延迟的数据互连。它与以太网等网络技术并列，但InfiniBand更注重于大规模、低延迟的数据传输，常用于服务器之间的互连和存储系统之间的互连。

启用 InfiniBand 需要设置：

```bash
export NCCL_IB_DISABLE=0
```

反之则设为 0。设为 0 通常更容易跑通，因为没有用到 NCCL。NCCL 是由NVIDIA开发的一个高效的并行通讯库，主要用于多GPU和分布式计算环境中的集体通信操作。它属于高性能并行计算库，特别是在深度学习、机器学习以及高性能计算（HPC）中非常常见。

多机之间的主要问题是机器找不到正确的网卡，导致通讯失败。所以需要禁用掉无关的网卡：

```bash
# ==== 网络接口过滤 ====
export NCCL_SOCKET_IFNAME=^lo
```

`^` 表示排除。这里需要视具体情况而定。建议先用 `ip addr` 命令查看一下每台机器的网口信息，确认哪些是有用的，哪些需要排除。

此外，还需要正确地配置网络，尤其是参数 `NCCL_IB_GID_INDEX` 和 `NCCL_IB_HCA`。具体的情况，需要根据机器自身配置来决定。

```bash
 ==== NCCL 通用设置 ====
export NCCL_DEBUG=WARN                         # 更高等级可设为 INFO 或 VERSION
export NCCL_ASYNC_ERROR_HANDLING=1             # 启用 NCCL 异步错误处理，提高健壮性
export NCCL_LAUNCH_MODE=GROUP                  # 减少通信延迟，推荐设置

# ==== IB/RoCE 网络优化 ====
export NCCL_IB_HCA=mlx5_0                      # 设置使用的网卡（你机器上有 mlx5_0 和 mlx5_1）
export NCCL_IB_GID_INDEX=0                     # 使用 IB/RoCE v1
export NCCL_IB_TC=106                          # 优先级设置，提升带宽服务质量
export NCCL_IB_TIMEOUT=22                      # 提高稳定性，避免大规模训练超时

# ==== GDR（GPU Direct RDMA） ====
export NCCL_NET_GDR_LEVEL=2                    # 强制使用 GPUDirect，如支持则显著提速

# ==== Ring 设置 ====
export NCCL_MIN_NRINGS=4                       # 建议设置为 4（NCCL 会自动适配实际拓扑）
```

### 获取节点信息

找到头节点、头节点的 IP 地址、尾节点以及节点列表等信息，在后续提交任务时可以自动化地复用变量。

```bash
NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
MASTER_NODE=$(head -n 1 <<< "$NODELIST")
MASTER_ADDR=$(tail -n 1 <<< "$NODELIST")
LAST_NODE=$(tail -n 1 <<< "$NODELIST")
NODE_NUM=($(echo $NODELIST | tr " " "\n" | wc -l))
NODE_COUNT=0
```

### 打印信息

将重要的信息打印在文件中，便于后续查看实验配置。

```bash
echo "NODE_NUM=$NODE_NUM"
echo "NODELIST:"
echo $NODELIST
echo "MASTER NODE=$MASTER_NODE, LAST_NODE=$LAST_NODE"
echo "MASTER ADDRESS=$MASTER_ADDR"
```

### 运行脚本

最后，我们需要在 NODE_LIST 中每个节点都提交任务。通过一个 for 循环实现：

```bash
for NODE in $NODE_LIST; do
	echo "run on $NODE, node_rank=$NODE_COUNT"
	if [ "$NODE" = "$LAST_NODE" ];then
		srun --nodes=1 --ntask=1 -w $NODE torchrun --nproc-per-node=8 --nnodes=$NODE_NUM \
			--node_rank=$NODE_COUNT --master_addr=$MASTER_ADDR --master_port=34567 \
			train.py
	else
		srun --nodes=1 --ntask=1 -w $NODE torchrun --nproc-per-node=8 --nnodes=$NODE_NUM \
			--node_rank=$NODE_COUNT --master_addr=$MASTER_ADDR --master_port=34567 \
			train.py &
	fi
	((NODE_COUNT++))
done
```

这里主要做的就是遍历节点列表，若不是尾节点，则悬挂任务（结尾为 `&`），并将 `NODE_COUNT` 加一；若为尾节点，则提交任务（结尾没有 `&`）。这样就可以将四个任务同时提交，建立互相之间的通信。那些用一行代码实现多节点的命令，我暂时没有成功过。

## 提交任务

最后用 sbatch 提交任务即可。

```bash
sbatch train.sh
```

