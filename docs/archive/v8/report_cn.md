# PTO运行时扩展：面向动态LLM推理的类型化工作负载调度方案

## 摘要

本报告针对大语言模型（LLM）推理中的动态性问题，提出PTO运行时扩展方案。通过**类型化工作负载表达**和**双并行模式**（数据并行 + CSP流水线并行），在保持静态图高效调度优势的同时，支持KV Cache变长、动态Batch、MoE稀疏路由等动态场景。

---

## 1. 问题与挑战

### 1.1 核心问题

| 问题 | 根因 | 影响 |
|------|------|------|
| **任务生成串行** | 单线程控制流 | 任务逐个枚举，阻塞并行性 |
| **执行队列串行** | 任务在单队列等待 | 无法重叠独立操作 |
| **调度固化** | 编程者无法控制dispatch/issue | 无法针对特定负载优化 |

**关键洞察**：这三个问题涉及任务**何时生成**、**在哪执行**、**以何顺序执行**。解决方案必须统一处理。

### 1.2 LLM推理的动态性

| 场景 | 动态性 | 静态图局限 |
|------|--------|------------|
| **KV Cache** | 不同用户序列长度不同 | 必须pad到最大长度，浪费带宽和显存 |
| **动态Batch** | batch内样本长度差异大 | pad到max(seq_len)，短序列浪费64倍计算 |
| **MoE路由** | 每token激活不同expert | 需为所有expert分配资源 |
| **TopK Attention** | 基于运行时分数选择 | 无法表达多路径选择逻辑 |

---

## 2. 设计原则

### 2.1 为什么选择Workload-Schedule范式

| 方面 | 原始任务图模型 | Workload-Schedule模型 |
|------|---------------|----------------------|
| **任务枚举** | 手动遍历循环 | `parallel_for`声明式枚举 |
| **依赖管理** | O(n²)手动连边 | 从结构自动推导 |
| **分发编程** | 手动拆分循环 | `dispatch_by(fn)` 一行代码 |
| **调度调整** | 重写任务图构建代码 | 仅修改Schedule参数 |
| **静态分析** | 运行时构建，分析有限 | 类型系统支持编译期检查 |

**核心价值**：

1. **任务枚举自动化** - `parallel_for`声明并行性，类型系统记录`Independent`依赖
2. **依赖结构化推导** - 嵌套结构隐式表达依赖，无需手动连边
3. **Schedule可编程** - `dispatch_by`、`stream_by`、`timing`提供完整调度控制
4. **优化迭代解耦** - 调整调度策略无需修改计算逻辑

### 2.2 为什么需要两种并行模式

| 模式 | 适用场景 | 示例 |
|------|----------|------|
| **数据并行** (`parallel_for`) | 相同操作处理不同数据 | Attention heads、MoE experts |
| **流水线并行** (CSP) | 不同操作并发执行 | Megakernel load/compute/store |

**原因**：单一抽象无法同时优雅表达两种模式。数据并行更简单，覆盖大多数场景；CSP处理复杂流水线。

---

## 3. 解决方案

### 3.1 类型化工作负载

```cpp
Workload<Axes, Task, Deps>  // 迭代空间、任务类型、依赖关系
```

**轴类型**（类型即约束）：

| 轴类型 | 语义 | 应用场景 |
|--------|------|----------|
| `Dense<N>` | 静态大小 | 固定头数 |
| `DenseDyn` | 运行时大小 | 动态batch |
| `Ragged` | 每元素长度不同 | 变长序列 |
| `Sparse` | CSR稀疏格式 | MoE路由 |

**依赖类型**（从结构推导）：

| 类型 | 含义 | 来源 |
|------|------|------|
| `Independent` | 所有任务可并行 | `parallel_for` |
| `Sequential` | 任务顺序执行 | `for_each`, `sequential()` |
| `Combined` | 调度决定顺序 | `combine()` |
| `ChannelDep` | 生产者-消费者 | CSP channels |

### 3.2 数据并行原语

```cpp
parallel_for(axis, body) → Workload[..., Independent]  // 独立并行
for_each(axis, body)     → Workload[..., Sequential]  // 顺序迭代
combine(w1, w2, ...)     → Workload[..., Combined]    // 组合（调度决定时序）
sequential(w1, w2, ...)  → Workload[..., Sequential]  // 显式顺序
select(sparse, body)     → Workload[..., Independent]  // 稀疏选择
cond(pred, then, else)   → Workload[...]               // 条件分支
task(kernel, params)     → Workload[Unit, Task, None]  // 叶子任务
```

**关键设计**：声明式原语（非命令式循环）使得：
- 并行任务枚举
- JIT分析和优化
- 从结构推导依赖

### 3.3 CSP流水线原语

**核心设计**：Channel传输**Workload**（包括Task），统一CSP与数据并行。

```cpp
Channel<Workload, N>   // 容量N的有界通道
process(name)          // 进程定义
  .consumes(channel)
  .produces(channel)
  .body(computation)   // 必须使用声明式原语
consume(channel, body) // 声明式消费（替代while循环）
Event = Channel<Signal, 0>  // 事件 = 无缓冲通道
```

**为什么Event = Channel<Signal, 0>**：统一同步模型。`record()` ≡ `send(e, Signal{})`，`wait()` ≡ `recv(e)`。

### 3.4 Schedule API

```cpp
auto schedule = workload.schedule()
    .dispatch(policy)      // 任务→核心分发
    .streams(n)            // 流数量
    .stream_by(fn)         // 任务→流分配
    .timing(policy);       // 发射时序
```

**为什么需要三层可编程**：

| 层次 | API | 控制内容 | 优化场景 |
|------|-----|----------|----------|
| Dispatch | `dispatch_by(fn)` | 任务→AICPU映射 | cache局部性、负载均衡 |
| Issue | `stream_by(fn)` | 任务→Stream分配 | 流水线效率 |
| Timing | `timing(policy)` | 发射时机 | eager/lazy/barrier |

**核心价值**：优化迭代无需修改Workload
```cpp
// 同一个workload
auto workload = create_attention_workload(batch, heads);

// 优化前：轮询调度
auto schedule_v1 = workload.schedule().dispatch(round_robin(16));

// 优化后：按batch分组（workload不变）
auto schedule_v2 = workload.schedule()
    .dispatch(affinity([](auto& t) { return t.param<int64_t>(0); }))
    .streams(4);
```

---

## 4. 关键问题的解决

### 4.1 任务并行生成

**问题**：控制流单线程导致任务串行生成

**解决**：声明式`parallel_for`允许并行枚举

```cpp
// 命令式（串行生成）
for (b = 0; b < batch; b++)
    for (i = 0; i < seq_len; i++)
        dispatch_task(b, i);  // 必须等内层循环结束

// 声明式（可并行枚举）
auto workload = parallel_for(DenseDyn(batch), [](int64_t b) {
    return for_each(DenseDyn(seq_len), [](int64_t i) {
        return task(kernel, {b, i});
    });
});
// enumerate()可并行生成所有任务
```

**原理**：`parallel_for`标记`Independent`，系统知道可以并行展开；`for_each`标记`Sequential`保持正确性。

### 4.2 跨图依赖管理

**问题**：多静态图任务同时运行，依赖管理困难

**解决**：基于工作负载类型的静态依赖分析 + Stream/Event同步

```cpp
auto layer = combine(
    rms_norm_workload,      // 先执行
    attention_workload,     // 依赖rms_norm
    ffn_workload            // 依赖attention
);
// 依赖关系由combine的顺序隐式表达
```

### 4.3 Human-In-The-Loop调度控制

**问题**：发现调度优化点后无法调整

**解决**：调度策略与工作负载正交

```cpp
// 调整前
auto schedule_v1 = workload.schedule().dispatch(round_robin(16));

// 调整后（仅修改dispatch策略，workload不变）
auto schedule_v2 = workload.schedule()
    .dispatch(affinity([](auto& t) { return t.param<int64_t>(0) % 4; }))
    .streams(4);
```

---

## 5. 实现与验证

### 5.1 实现架构

```
include/pto/rt/
├── types.hpp          # 核心类型：Axis, Task, Tensor
├── workload.hpp       # Workload<Axes, Task, Deps>
├── primitives.hpp     # parallel_for, for_each, select
├── schedule.hpp       # Schedule, DispatchPolicy
├── csp.hpp            # Channel, Process, Pipeline
└── cpu/simulation.hpp # CPU仿真
```

### 5.2 PTO-ISA集成

所有内核使用原生PTO-ISA tile操作：

| 内核 | PTO-ISA指令 |
|------|-------------|
| Attention | TLOAD, TMATMUL, TTRANS, TROWMAX, TEXP, TROWSUM, TSTORE |
| RMSNorm | TLOAD, TMUL, TROWSUM, TRSQRT, TROWEXPANDMUL, TSTORE |
| FFN | TMATMUL (gate, up, down), SiLU, TMUL |

### 5.3 DeepSeek-V3.2-Exp验证

| 测试 | 配置 | 结果 |
|------|------|------|
| 变长Attention | 4 batch × 8 heads = 32 tasks | PASS |
| MoE稀疏路由 | 2048 shared + 16384 routed = 18432 tasks | PASS |
| 完整前向 | 2 layers, 9 tasks/layer | PASS |
| flash_attention_demo | 最大误差 3.26e-09 | PASS |

---

## 6. 对比分析

| 特性 | 静态图+日历调度 | 动态图+AICPU | **PTO RT Extension** |
|------|----------------|--------------|---------------------|
| 变长序列 | 需padding | 支持 | Ragged轴类型 |
| MoE稀疏路由 | 全Expert分配 | 支持 | select原语 |
| 任务并行生成 | 编译期 | 串行 | parallel_for |
| Dispatch编程 | 编译期固化 | 运行时手动 | `dispatch_by(fn)` |
| Issue编程 | 日历固定 | 各AICPU独立 | `stream_by(fn)` |
| 优化迭代 | 重新编译 | 重写代码 | 仅修改Schedule |

---

## 7. 结论

### 主要贡献

1. **类型化工作负载**：`Workload<Axes, Task, Deps>`声明式表达动态LLM推理结构
2. **双并行模式**：数据并行（parallel_for, select）+ CSP流水线（Channel, Process）
3. **灵活调度API**：dispatch、streams、timing实现Human-In-The-Loop调度控制
4. **PTO-ISA兼容**：所有内核使用原生PTO tile操作

### 未来工作

1. **NPU后端**：迁移到Ascend NPU验证实际性能
2. **JIT编译**：workload到日历调度的JIT编译优化
3. **自动调度**：基于cost model自动选择dispatch/stream策略
4. **分布式扩展**：多设备工作负载分发

---

*报告生成日期：2025-01-18*
