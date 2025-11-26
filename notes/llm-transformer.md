# 大模型核心：Transformer 架构与 LoRA 微调

## 一、 Transformer：Encoder 与 Decoder 的本质区别

在面试中，仅仅回答“一个是编码一个是解码”是不够的。你需要从**注意力机制**和**应用场景**两个维度进行深入阐述。

### 1. 架构对比表

| 特性 | Encoder (编码器) | Decoder (解码器) |
| :--- | :--- | :--- |
| **核心机制** | **双向注意力 (Bidirectional)** | **因果/单向注意力 (Causal / Masked)** |
| **可视范围** | 能看到全句（过去和未来） | 只能看到当前词之前的历史（不能偷看答案） |
| **典型代表** | BERT, RoBERTa, ViT | GPT 系列, LLaMA, DeepSeek |
| **擅长任务** | 理解任务（情感分析、命名实体识别） | 生成任务（写小说、写代码） |
| **并行性** | 训练和推理都可以并行 | 训练可并行，**推理必须串行**（依赖 KV Cache） |

### 2. 关键组件：Cross Attention (交叉注意力)

这是 Transformer 中连接“理解”与“生成”的桥梁。很多面试官喜欢问 Q、K、V 的来源。

**数学公式：**

$$Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

**数据流向（灵魂拷问）：**
* **Query ($Q$)：来自 Decoder**。代表“我现在想预测下一个词，我需要查询什么信息？”（解码器上一层的输出）。
* **Key ($K$)：来自 Encoder**。代表“原文输入中每个词的索引特征”（编码器最后一层的输出）。
* **Value ($V$)：来自 Encoder**。代表“原文输入中每个词的具体内容特征”（编码器最后一层的输出）。

**PyTorch 伪代码实现：**

```python
import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.linear_q = nn.Linear(d_model, d_k) # Q 的线性变换
        self.linear_k = nn.Linear(d_model, d_k) # K 的线性变换
        self.linear_v = nn.Linear(d_model, d_k) # V 的线性变换

    def forward(self, x_decoder, x_encoder):
        # Q 来自解码器 (Batch, Seq_Dec, Dim)
        Q = self.linear_q(x_decoder)
        # K, V 来自编码器 (Batch, Seq_Enc, Dim)
        K = self.linear_k(x_encoder)
        V = self.linear_v(x_encoder)
        
        # 计算注意力分数: Q * K转置 / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Softmax 归一化
        attn = torch.softmax(scores, dim=-1)
        
        # 加权求和: Attention * V
        output = torch.matmul(attn, V)
        return output
```
# 二、LoRA (Low-Rank Adaptation) 原理与实战

全量微调（Full Fine-tuning）成本太高，LoRA 是目前性价比最高的 PEFT（参数高效微调）方案，广泛应用于大模型微调。

## 1. 核心原理：旁路更新

LoRA 的核心思想是冻结预训练好的模型权重，并假设模型参数的更新量是**低秩（Low-Rank）**的。我们在原始权重 $W_0$ 旁增加两个小矩阵 $A$ 和 $B$ 来模拟参数更新。

* **前向传播公式：**
    $$h = W_0 x + \Delta W x = W_0 x + B A x$$
    
* **维度变化：**
    * $W_0$: $[d, d]$ (比如 4096 x 4096，参数量巨大，**冻结不更新**)
    * $A$: $[r, d]$ (降维矩阵，r 通常为 8 或 16)
    * $B$: $[d, r]$ (升维矩阵)
    * **优势：** 训练参数量通常仅为原模型的 1% 甚至更低，且推理时可以将 $BA$ 乘积加回 $W_0$，实现零推理延迟。

## 2. 初始化策略（面试必考坑点）

为了保证训练刚开始时，模型的效果**不下降**（行为等价于原预训练模型），LoRA 对两个矩阵的初始化非常讲究：

* **矩阵 A：** 使用 **高斯分布 (Kaiming / Normal)** 初始化。
    * *目的：* 引入随机性，打破对称，让网络能从不同的方向开始学习特征。
* **矩阵 B：** 使用 **全 0 (Zero)** 初始化。
    * *目的：* 使得初始状态下 $\Delta W = B \times A = 0 \times A = 0$。
    * *结果：* 初始输出 $h = W_0 x + 0$，这保证了 LoRA 模块接入瞬间，模型输出与原模型完全一致，训练极其稳定，不会因为引入随机噪声导致模型能力瞬间崩塌。

## 3. 代码实战（使用 huggingface/peft 库）

这是实际工作中调用 LoRA 的标准写法：

```python
from peft import LoraConfig, get_peft_model, TaskType

# 1. 定义 LoRA 配置
config = LoraConfig(
    r=16,               # 秩 (Rank)，越大能学的特征越多，但显存占用越大
    lora_alpha=32,      # 缩放系数，通常设置为 r 的 2 倍，用于稳定训练
    target_modules=["q_proj", "v_proj"], # 指定只微调 Attention 层中的 Q 和 V 矩阵
    lora_dropout=0.05,  # 防止过拟合
    bias="none",        # 通常不微调偏置项
    task_type=TaskType.CAUSAL_LM 
)

# 2. 加载模型并应用 LoRA
# model 是你加载好的 transformers 模型 (如 LLaMA)
model = get_peft_model(base_model, config)

# 3. 打印可训练参数量对比
model.print_trainable_parameters() 
# 输出示例：trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
# 可以看到仅需训练 0.06% 的参数