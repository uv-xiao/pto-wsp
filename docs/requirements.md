# Runtime 问题

## 概念对齐

### 静态图与动态图

在深度学习框架中，计算图的执行方式主要分为两种：静态图（Static Graph）和动态图（Dynamic Graph）。

#### 静态图（Static Graph）

静态图在执行前需要先构建完整的计算图结构，然后一次性编译和优化，最后再执行。主要特点包括：

- **编译时优化**：在构建阶段可以进行图级别的优化，如算子融合、内存优化等
- **执行效率高**：编译后的图执行效率通常更高，适合生产环境
- **调试困难**：由于图在执行前已确定，调试和错误定位相对困难
- **灵活性较低**：无法在运行时动态改变图结构

1. 静态图的调度可以采用预先排列好的方式（我们称之为**日历调度**，calendar scheduling）。

    - 日历调度的核心思想是：在编译期就把所有算子、所有批次、各类资源、数据流的执行顺序以及调度计划全部排好，就像制定一份确定的时间表（即“日历”）。
    - 这样做的好处是可以最大化利用算力和内存带宽、减少运行时的调度开销，并可以高效地进行批量算子融合、pipeline 等各种优化。
    - 日历调度广泛用于硬件推理引擎（如 Ascend、TPU、NPU 内的 Graph Engine）、以及部分模型训练系统（如一部分分布式大模型训练 pipeline）。
    - 典型实现：所有 task、kernel、memory address、数据流和同步点都静态地“编排好”，整个大图以调度表驱动执行，运行期仅需“按表同步”。
    - 局限性：任何运行时 shape 变化、动态 batch、动态条件分支等情况都极难支持。因为编译出的调度日历已经固化，变化会造成调度/地址/内存的混乱。

    日历调度适合应用于大批量、任务高度一致、shape 恒定的大模型推理，追求极致的吞吐率和稳定性。但对于需支持动态变长序列、灵活样本组、部分在线推理（如 KV-Cache、变长 batch）的场景，则常需要动态（runtime）调度能力。

2. 静态图的典型限制示例：无法处理不同序列长度的 KV Cache。静态图设计时，需要提前确定所有张量的尺寸、执行流程。这导致它在下面的应用中特别不适用：

    - **序列长度（seq len）变化的场景**
    例如在大模型推理中，KV Cache（Key/Value Cache）通常用于加速自回归生成，但用户每次的输入序列长度（以及历史长度）可能各不相同。
    静态图必须在编译期就固定 KV Cache 张量的 shape，导致只能支持定长输入。如果希望不同样本不同长度、或生成过程中动态增长，就难以用静态图天然支持。
    - **典型现象**
        - 有些静态图推理引擎要求用户 pad 到固定最大长度，使计算和显存浪费严重
        - 动态 batch、变长序列/kv-cache 需求难以支持


#### 动态图（Dynamic Graph）

动态图在运行时即时构建和执行计算图，每次执行都会重新构建图。主要特点包括：

- **开发友好**：代码编写直观，易于调试，支持 Python 原生控制流
- **灵活性高**：可以在运行时根据数据动态调整计算流程
- **执行开销**：每次执行都需要重新构建图，存在一定的运行时开销
- **优化受限**：难以进行全局的图级别优化

在动态图（Dynamic Graph）模式下，运行时系统需要具备与静态图完全不同的支撑能力，其核心要求包括：

#### 1. 依赖管理与调度

- **动态依赖解析**
  动态图在执行时根据实际数据和控制流即时构建算子依赖关系。运行时必须能够实时追踪并解析操作之间的依赖关系（或称“前驱-后继”），以保证执行的正确性。
- **算子调度与即时执行**
  动态图通常采用“定义即执行”策略（define-by-run），每一步的计算结果可被立即用于后续操作，算子调度和调优往往在每次前向执行过程中实时完成。

#### 2. 资源与内存管理

- **即时内存分配与回收**
  由于计算流程和中间张量的 shape、生命周期在运行期才确定，运行时系统需即时为每个新建的 Tensor 分配内存，并及时回收不再被引用的内存（参考 PyTorch、TensorFlow Eager Execution 的内存管理）。


## 当前LLM中常见的动态性

在现代大模型推理与训练中，“动态性”主要集中体现在以下几个典型应用场景或技术方案：

### 1. KV Cache长度动态变化
KV Cache（Key/Value Cache）常用于Transformer自回归推理以缓存历史token的Key、Value，从而高效生成下一步输出。但每个用户推理/训练时，输入序列长度和上下文（历史token数）往往不固定。这导致：

- 不同样本的KV Cache张量shape不同（如[batch, cur_len, dim]）。
- 生成过程中每步动态扩展KV Cache内容，需要能支持动态写入和变更shape。
- 静态图很难直接表达或管理这个"变长"特性，通常需强行padding成最大长度（带来巨大的带宽和显存浪费），或采用多编译版本，导致可维护性和效率下降。
- 动态图/Runtime调度可按需分配/拼接KV Cache，灵活支持变长输入和高效推理。

**伪代码示例：**

```python
# 动态图支持：KV Cache 按需扩展
def GenerateWithKVCache(model, input_ids, max_new_tokens):
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]  # 不同样本的初始长度可能不同

    # 初始KV Cache：shape = [batch, seq_len, hidden_dim]
    k_cache = None
    v_cache = None

    for step in range(max_new_tokens):
        # 当前输入：shape = [batch, 1] (单个新token)
        current_input = input_ids[:, -1:] if step == 0 else next_token

        # 计算当前step的K, V
        k_new, v_new = model.attention.compute_kv(current_input)

        # 动态拼接：KV Cache长度在运行时增长
        if k_cache is None:
            k_cache = k_new  # 首次：shape = [batch, 1, hidden_dim]
            v_cache = v_new
        else:
            # 动态扩展：shape从 [batch, cur_len, hidden_dim]
            # 变为 [batch, cur_len+1, hidden_dim]
            k_cache = concat([k_cache, k_new], dim=1)
            v_cache = concat([v_cache, v_new], dim=1)

        # 使用完整的KV Cache计算attention
        output = model.attention(q=current_input, k=k_cache, v=v_cache)
        next_token = model.generate_next_token(output)

    return generated_sequence

# 静态图的限制：必须预先确定最大长度
# 所有样本必须pad到max_seq_len，造成浪费
# k_cache_static = zeros([batch, max_seq_len, hidden_dim])  # 大量零填充
```

```python
# 不同样本的初始序列长度不同
batch_samples = [
    [1, 2, 3, 4],           # seq_len = 4
    [10, 20, 30, 40, 50],   # seq_len = 5
    [100, 200]              # seq_len = 2
]

# 动态图：每个样本的KV Cache按实际长度分配
# 静态图：必须pad到max_len=5，样本3浪费3个位置的内存和计算
```

### 2. 动态 Batch 长度变化举例

假设我们有一个 batch，其中每个样本的序列长度都不一样，这在推理/生成场景下非常常见。例如：

```python
# 不同 batch 样本的真实序列长度
actSeqKey = [512, 2048, 8192, 32768]  # batch size = 4，每条样本长度差异巨大

# 动态图下，每个样本按实际长度分配和处理
for bIdx, curSeq in enumerate(actSeqKey):
    # 以 batch 的实际序列长度动态决定 KV Cache 和注意力计算逻辑
    # 例如动态扩展 KV Cache，或者每个样本循环内实际处理的 block 数量各不相同
    print(f"Batch {bIdx}: 实际序列长度 = {curSeq}")
    # ...后续真实逻辑...

# 静态图下，通常只能 pad 到最大 batch 长度，导致大量计算和内存浪费
maxSeq = max(actSeqKey)
# 假设 hidden_dim = 4096
k_cache_static = zeros([len(actSeqKey), maxSeq, 4096])  # 大部分位置其实是没用的padding
```

在动态 TopK 索引/稀疏注意力中，不同 batch 的处理路径和循环体次数也会根据 actSeqKey 动态变化，必须 runtime 决定。

### 3. Lightning Indexer TopK Attention（DeepSeek v3.2）
DeepSeek v3.2 中的 Lightning Indexer 采用 TopK 稀疏注意力机制，其核心思想是：
- 对于每个 query token，从所有 key tokens 中动态选择 top-k（通常 k=2048）个最相关的 key 位置进行 attention 计算。
- 通过计算 query 和 key 的相似度分数，应用权重和 ReLU 激活后，动态选择 top-k 个位置，而非计算全对全的密集 attention。
- 在因果推理场景下，每个位置的"有效序列长度"（effSeq）在运行时动态确定，导致需要计算的 block 数量和 TopK 的处理路径都不同。
- 静态图难以处理这种基于运行时相似度分数的动态 TopK 选择，以及不同序列长度下的多路径处理逻辑。

**伪代码示例：**

```python
# DeepSeek v3.2 Lightning Indexer TopK Attention
# 动态图支持：根据运行时相似度分数动态选择 TopK
def LightningIndexerTopK(query, key, weights, actSeqKey, blockTable, selectedCount=2048):
    """
    query: [B, S1, indexN1, indexD] - 当前 batch 的 query tokens
    key: [blockNum, blockSize, n2, indexD] - 缓存的 key blocks
    weights: [B, S1, indexN1] - query 的权重
    actSeqKey: [B] - 每个 batch 的实际序列长度（动态）
    blockTable: [B, maxBlockNum] - 每个 batch 的 block 索引表
    selectedCount: 2048 - 每个 query 位置选择的 top-k key 数量
    """
    B, S1, indexN1, indexD = query.shape
    blockNum, blockSize, n2, _ = key.shape

    topkRes = zeros([B, S1, n2, selectedCount], dtype=int32)

    # 遍历每个 batch
    for bIdx in range(B):
        curSeq = actSeqKey[bIdx]  # 运行时获取：该 batch 的实际序列长度

        # 遍历每个序列位置（因果推理：从后往前）
        for s1Idx in range(S1):
            # 因果推理：动态计算有效序列长度
            # 位置越靠后，能看到的序列越长
            casualOffset = S1 - s1Idx - 1
            effSeq = curSeq - casualOffset  # 运行时动态确定

            # 动态计算需要处理的 block 数量
            actBlock = (effSeq + blockSize - 1) // blockSize

            # 遍历每个 index head
            for n2Idx in range(n2):
                # 计算 query 与所有相关 key blocks 的相似度
                localSum = zeros([effSeq])  # 动态长度

                # 动态循环：只处理有效的 blocks
                for blockIdx in range(actBlock):
                    # 从 blockTable 动态获取实际的 block 索引
                    curBlockIdx = blockTable[bIdx, blockIdx]

                    # 计算 query 与当前 block 的 key 的相似度
                    curQ = query[bIdx, s1Idx, n2Idx*group:(n2Idx+1)*group, :]  # (group, indexD)
                    curK = key[curBlockIdx, :blockSize, n2Idx, :]  # (blockSize, indexD)

                    # 矩阵乘法：计算相似度分数
                    mmRes = matmul(curQ, transpose(curK))  # (group, blockSize)

                    # 应用权重和 ReLU
                    curW = weights[bIdx, s1Idx, n2Idx*group:(n2Idx+1)*group]  # (group,)
                    reluRes = relu(mmRes)  # (group, blockSize)
                    mulRes = reluRes * curW.unsqueeze(-1)  # (group, blockSize)
                    sumRes = sum(mulRes, dim=0)  # (blockSize,) - 每个 key 位置的分数

                    # 累积到 localSum（动态位置）
                    startPos = blockIdx * blockSize
                    endPos = min(startPos + blockSize, effSeq)
                    localSum[startPos:endPos] = sumRes[:endPos-startPos]

                # 根据 effSeq 动态选择 TopK 处理路径
                # 不同长度需要不同的 pad 和 TopK 策略
                if effSeq <= 2048:
                    # 短序列：pad 到 2K
                    padX = pad(localSum, pad_length=2048, pad_value=-FLT_MAX)
                    topkValues, topkIndices = topk(padX, k=selectedCount, dim=-1)
                    # 只保留有效部分
                    topkRes[bIdx, s1Idx, n2Idx, :effSeq] = topkIndices[:effSeq]
                    topkRes[bIdx, s1Idx, n2Idx, effSeq:] = -1  # pad

                elif effSeq <= 8192:
                    # 中等序列：pad 到 8K
                    padX = pad(localSum, pad_length=8192, pad_value=-FLT_MAX)
                    topkValues, topkIndices = topk(padX, k=selectedCount, dim=-1)
                    topkRes[bIdx, s1Idx, n2Idx, :] = topkIndices

                elif effSeq <= 65536:
                    # 长序列：pad 到 64K
                    padX = pad(localSum, pad_length=65536, pad_value=-FLT_MAX)
                    topkValues, topkIndices = topk(padX, k=selectedCount, dim=-1)
                    topkRes[bIdx, s1Idx, n2Idx, :] = topkIndices

                else:
                    # 超长序列：pad 到 128K
                    padX = pad(localSum, pad_length=128*1024, pad_value=-FLT_MAX)
                    topkValues, topkIndices = topk(padX, k=selectedCount, dim=-1)
                    topkRes[bIdx, s1Idx, n2Idx, :] = topkIndices

    return topkRes  # [B, S1, n2, selectedCount] - 每个 query 位置选中的 key 索引

# 关键动态性：
# 1. actSeqKey[bIdx] - 每个 batch 的序列长度在运行时不同
# 2. effSeq = curSeq - casualOffset - 因果推理中每个位置的有效长度不同
# 3. actBlock = (effSeq + blockSize - 1) // blockSize - 动态 block 数量
# 4. TopK 路径选择：根据 effSeq 动态选择 2K/8K/64K/128K 处理路径
# 5. topkRes 中选中的 key 位置完全由运行时相似度分数决定
```

### 4. MoE（Mixture-of-Experts）动态路由
MoE模型通过路由机制为每个输入token"动态"选择部分Expert（子模型）进行独立计算：
- 每个batch、每个token根据门控网络（Gater）的结果，分配到不同的Experts（通常是稀疏激活的）。
- 被激活的Expert数量、token分组和实际触发的计算负载都在运行时决定。
- 需要在Runtime进行token到Expert的scatter/gather，并动态调度各Expert的并发任务以及合并输出。
- 静态图对于MoE动态路由（特别是token动态分组与Expert动态激活）支持极不灵活，需要最大化分配/调度所有Expert分支，效率极低。
- 动态图天然适合表达和管理这种基于运行时输入"稀疏激活"机制的复杂流程。

**伪代码示例：**

```python
# 动态图支持：MoE路由在运行时决定激活哪些Expert
def MoELayer(x, experts, gating_network, top_k=2):
    """
    x: [batch, seq_len, hidden_dim]
    experts: List[Expert], 例如8个Expert
    top_k: 每个token激活k个Expert（通常k=2，稀疏激活）
    """
    batch_size, seq_len, hidden_dim = x.shape
    num_experts = len(experts)

    # 门控网络：为每个token计算所有Expert的权重
    # gate_logits: [batch, seq_len, num_experts]
    gate_logits = gating_network(x)

    # 运行时动态选择top_k个Expert（每个token可能选择不同的Expert组合）
    gate_probs, selected_experts = top_k(gate_logits, k=top_k)
    # selected_experts: [batch, seq_len, top_k] - 每个token选中的Expert ID

    # 动态分组：将token按Expert分组
    # 不同batch、不同token可能激活完全不同的Expert组合
    expert_to_tokens = {}  # {expert_id: List[(batch_idx, token_idx, weight)]}

    for b in range(batch_size):
        for t in range(seq_len):
            for k_idx in range(top_k):
                expert_id = selected_experts[b, t, k_idx]
                weight = gate_probs[b, t, k_idx]

                if expert_id not in expert_to_tokens:
                    expert_to_tokens[expert_id] = []
                expert_to_tokens[expert_id].append((b, t, weight, x[b, t]))

    # 动态调度：只为被激活的Expert执行计算
    outputs = zeros([batch_size, seq_len, hidden_dim])

    for expert_id, tokens in expert_to_tokens.items():
        # 收集该Expert需要处理的token
        expert_inputs = [token_data for _, _, _, token_data in tokens]
        expert_inputs = stack(expert_inputs)  # 动态shape

        # 执行Expert计算（只计算被激活的Expert）
        expert_outputs = experts[expert_id](expert_inputs)

        # 动态scatter：将结果写回对应位置
        for idx, (b, t, weight, _) in enumerate(tokens):
            outputs[b, t] += weight * expert_outputs[idx]

    return outputs

# 示例：不同token激活不同Expert
# Token 0: 激活 Expert[1, 3] -> 只计算Expert1和Expert3
# Token 1: 激活 Expert[2, 5] -> 只计算Expert2和Expert5
# Token 2: 激活 Expert[1, 7] -> 只计算Expert1和Expert7
# 静态图需要为所有8个Expert都分配资源，动态图只计算实际需要的
```

## PyPTO 的采用模式的演进

### 0. xxx

### 1. Host跑控制流，不断产生静态图（不含调度），静态图交给aicpu动态调度执行

问题：调度可能不是最佳 -> 日历调度

### 2. Host跑控制流，不断产生静态图（日历调度），aicore按日历调度执行（只仿真实现，未上板实现）

问题：
1. 认为KernelLaunch代价高
2. 有些控制流依赖计算产生的tensor的value

-> aicpu执行控制流

### 3. aicpu跑控制流，不断产生静态图（不含调度），交给aicpu动态调度执行

问题：静态图粒度小，优化不充分 -> Function Concat & Stitch

### 4. aicpu跑控制流并stitch，不断产生更大的计算图（不含调度），交给aicpu动态调度执行

注：至此，已经无法使用日历调度，因为一张图内的task个数都会动态变化。

问题：for循环次数多时，单迭代优化效果差，且stitch压力大 -> 多档位，化 O(N) 的动态性为 O(logN) 的动态性

### 5. host完成多档位图的优化，aicpu跑控制流并stitch，不断产生更大的计算图（不含调度），交给aicpu动态调度执行

## 关键问题

### 控制流单线程，导致任务生成串行

```py
for b in loop(batch): # within this loop, task are independent
    for i in loop(seq_len): # within this loop, task are dependent
        xxx
```

上述是一个典型attention算子逻辑，在当前的runtime中，只有等内层 loop 执行结束后，外层 loop 才能产生任务，导致并行性差。

提问：硬件只有一个aicpu来生成任务，怎么并行生成任务？

### 串行执行队列

一个典型 pageattention 的静态图形如

```txt
时间 →  ───────────────────────────────────────────────>

CUBE :  ┌───────┐  ┌───────┐  ┌───────┐
        │ cube1 │  │ cube2 │  │ cube3 │
        └───────┘  └───────┘  └───────┘
             │         │         │
             ▼         ▼         ▼
VECTOR:      ┌───────┐  ┌───────┐  ┌───────┐
             │vector1│─>│vector2│─>│vector3│
             └───────┘  └───────┘  └───────┘
```

一般来说，cube和vector由于有依赖，会产生一些空隙。这些空隙可以
1. 通过 stitch 不同 batch 的任务来部分掩盖。（管理好依赖再开始执行）

   通过stitch掩盖存在明显的瓶颈，stitch越多任务性能越好，但是stitch代价越大，存在明显的头开销。
   - 一种规避方式是，先下发一个任务，在该任务执行时stitch，stitch后续任务，这样越晚执行的任务性能越好，存在一个冷启动无法避免。
   - 即使拼接的很大，也只能缓解这个问题，无法从根本上解决。

2. 通过允许多个在队列中的静态图任务同时运行来掩盖。（边执行边管理依赖）

   该方案还未实现。问题是，一个已经进队列的静态图任务（代码+数据）和其他任务之间的依赖关系的建立和管理难度较大。尤其是当考虑已经完成部分子任务执行的静态图和后续静态图。

   注：通过执行异步dispatch任务的方式捕获依赖，在面对读后写，写后写等场景需要额外的机制来实现，并不能仅依靠计算逻辑和数据依赖。这部分还是有点难度，所以NV搞了一套 Stream/Event，把这个问题交给用户显示声明。

### 无法 Human-In-The-Loop 调整调度策略

下图是每行表示一个任务，每列排列同一个batch，发现这种行为不好，因为每个batch各自有一个冷启动，如果改成每行一个Batch，性能会提升10%。但是即使发现了这一点，也无法在aicpu调度（通用）逻辑上调整（专用）。

```txt
时间 →  ───────────────────────────────────────────────────────────────>

Core0:  B0_T0  ──>  B1_T0  ──>  B2_T0  ──>  B3_T0  ──>  B4_T0
Core1:  B0_T1  ──>  B1_T1  ──>  B2_T1  ──>  B3_T1  ──>  B4_T1
Core2:  B0_T2  ──>  B1_T2  ──>  B2_T2  ──>  B3_T2  ──>  B4_T2
Core3:  B0_T3  ──>  B1_T3  ──>  B2_T3  ──>  B3_T3  ──>  B4_T3
Core4:  B0_T4  ──>  B1_T4  ──>  B2_T4  ──>  B3_T4  ──>  B4_T4
```

每行排列同一个batch：

```txt
时间 →  ───────────────────────────────────────────────────────────────>

Core0:  B0_T0  ──>  B0_T1  ──>  B0_T2  ──>  B0_T3  ──>  B0_T4
Core1:  B1_T0  ──>  B1_T1  ──>  B1_T2  ──>  B1_T3  ──>  B1_T4
Core2:  B2_T0  ──>  B2_T1  ──>  B2_T2  ──>  B2_T3  ──>  B2_T4
Core3:  B3_T0  ──>  B3_T1  ──>  B3_T2  ──>  B3_T3  ──>  B3_T4
Core4:  B4_T0  ──>  B4_T1  ──>  B4_T2  ──>  B4_T3  ──>  B4_T4
```

这里想表达的核心是，当前Runtime Human-In-The-Loop的能力弱。

## 易用性问题

### Rumtime 接口（DevAscendProgram） Human 不友好，手写 kernel 后，无法手写 RootFunction （静态图） 并执行

### Machine和CostModel没有共代码

## 问题提炼

问题可以简单抽象为

1. 硬件模型 1A+16B，1B=4C+24D+48E的架构，如果不考虑分布式，简化一下就是，1A+4B+24C+48D的模型，A、B是cpu，C、D是非常弱的cpu+DSA（SIMD）扩展。A -> B 3u, A -> C/D 3u, B -> C/D 0u。
2. 软件模型，任务从A下发，跑在C、D上，有很多C、D上的kernel，如何控制A、B，组装这些kernel高性能完成一个任务。简化一下，我们只考虑用户脑海中有一种调用和排布方式（日历调度+动态生成任务的时候用户知道怎么排列任务，该用户可以是人、可以是PyPTO框架的Pass部分），我们怎么提供接口让他描述，并且帮他执行这些kernel。
   1. 对于单个kernel而言，怎么定调用接口，怎么给它准备参数。
   2. 对于多个kernel而言，怎么表达图（即kernel的并行、依赖关系），怎么分配内存，怎么调度。

已经尝试过的方案的简单总结：
1. 纯静态图，可以日历调度（可以Human-In-The-Loop Runtime），但是处理不了动态shape
2. 动态控制流下发静态图，要
   1. 解决并行下发的问题
   2. 解决两个静态图一起跑的问题
      1. 通过静态拼接图的方式，头开销
      2. 通过执行下发产生任务，依赖管理难度大
   3. 如何细粒度Human-In-The-Loop Runtime行为
3. 最好还能支持
   1. 人手写 aicore kernel + rootfunction/task generate code。
   2. CostModel