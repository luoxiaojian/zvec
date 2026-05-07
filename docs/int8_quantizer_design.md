# Int8 Quantizer 设计文档

## 一、Uniform Int8 Quantizer

### 1. 概述

Uniform Int8 Quantizer 采用**全局统一的 scale/bias** 将 float32 向量量化为 int8 向量，使得 L2 距离可以完全在整数域中计算，无需逐向量反量化。

### 2. 使用方式

| 组件 | 注册名 | 说明 |
|------|--------|------|
| Converter | `UniformInt8StreamingConverter` | 训练阶段计算全局 scale/bias，并量化数据 |
| Reformer | `UniformInt8StreamingReformer` | 查询/记录 float→int8 在线量化 |
| Metric | `UniformInt8` | 使用 int8 L2 距离计算 |

**关键参数（自动计算，无需手动设置）：**

- `uniform_int8.reformer.scale` — 全局缩放因子
- `uniform_int8.reformer.bias` — 全局偏移量
- `proxima.uniform_int8.metric.origin_metric_name` — 原始距离类型（目前仅支持 `SquaredEuclidean`）

**Turbo 层枚举：** `turbo::QuantizeType::kUniform`

### 3. 适用范围

| 维度 | 支持范围 |
|------|----------|
| 数据集 | 任意 float32 向量数据集（值域不限） |
| 量化目标类型 | `DT_INT8`（输出维度 = 原始维度） |
| 距离类型 | `SquaredEuclidean`（int8 域 L2，后乘 1/scale² 还原真实 L2） |
| 硬件要求 | AVX-512 VNNI（无 VNNI 时回退到标量路径） |

### 4. 实现思路

#### 4.1 训练（Converter::train）

1. 遍历全部训练向量，统计 `global_min` / `global_max`
2. 快速路径：若所有值为整数且 range ≤ 254 → `scale = 1`（无损量化）
3. 否则线性缩放：`scale = 254 / range`，`bias = -global_min × scale - 127`
4. 将 scale/bias 写入 reformer_params 和 converter_params 持久化

#### 4.2 量化公式

查询和数据端采用**完全相同**的量化公式：

```
正向：int8_val = clamp(round(float_val × scale + bias), -127, 127)
反向：float_val ≈ (int8_val - bias) / scale
```

#### 4.3 距离计算（AVX-512）

```
对每 32 个元素：
  1. 加载 int8 → 符号扩展 int16 (vpmovsx)
  2. 差值 diff = query - vec (vpsubw)
  3. 差值平方 + 邻对求和 (vpmaddwd) → int32
  4. 累加 (vpaddd)
最终 reduce 为单个 int32 → cast to float
```

距离关系：`int8_L2 = scale² × real_L2`，通过 Reformer 的 `normalize()` 乘以 `1/scale²` 还原。

#### 4.4 向量存储布局

```
数据端：[ dim × int8 ]
查询端：[ dim × int8 ]（与数据端对称，无 extra fields）
```

---

## 二、SIFT Int8 Quantizer

### 1. 概述

SIFT Int8 Quantizer 专为 **SIFT 类数据集**设计。SIFT 特征向量的值是非负整数（通常 [0, 218]），天然适合 scale=1 的无损量化。

**核心创新**：参考 Record Quantizer 的 extra fields 思路，将每个向量的**平方和的一半**（`sq_sum / 2.0f`）作为尾部附加字段存储，使距离计算可以利用 **dpbusd 指令**（uint8 × int8 内积）替代传统的逐元素差值平方求和，获得显著的计算性能优势。

### 2. 使用方式

| 组件 | 注册名 | 说明 |
|------|--------|------|
| Converter | `SiftInt8StreamingConverter` | 训练阶段计算 bias，量化数据并附加 sq_sum_half |
| Reformer | `SiftInt8StreamingReformer` | 查询量化为 uint8，记录量化为 int8 + sq_sum_half |
| Metric (add 路径) | `SiftInt8` | 图构建阶段使用对称 int8×int8 L2 距离 |
| Metric (search 路径) | `SiftInt8Query` | 查询阶段使用 dpbusd 不对称距离（`SiftInt8::query_metric()` 返回） |

**关键参数（自动计算，无需手动设置）：**

- `sift_int8.reformer.bias` — 偏移量（将值域映射到 int8 范围）
- `proxima.sift_int8.metric.origin_metric_name` — 原始距离类型（目前仅支持 `SquaredEuclidean`）

**Turbo 层枚举：** `turbo::QuantizeType::kSift`

### 3. 适用范围

| 维度 | 支持范围 |
|------|----------|
| 数据集 | SIFT 类数据集：非负整数值，值域在 [0, 255] 内 |
| 量化目标类型 | `DT_INT8`（输出维度 = 原始维度 + 4 字节 extra field） |
| 距离类型 | `SquaredEuclidean`（单调排序代理，非真实 L2 值） |
| 硬件要求 | AVX-512 VNNI（dpbusd 指令） |

> **适用约束：** 此量化器假设 scale=1，仅适用于值域紧凑且为非负整数的数据集。对于值域超过 [0, 255] 或包含浮点值的数据集，应使用 Uniform Int8 Quantizer。

### 4. 实现思路

#### 4.1 训练（Converter::train）

1. 遍历全部训练向量，统计 `global_min`
2. 计算 `bias = -round(global_min) - 128`（将 global_min 映射到 int8 的 -128）
3. 将 bias 写入 reformer_params 持久化

与 Uniform Quantizer 不同，**无需计算 scale**（固定为 1），训练过程更简洁。

#### 4.2 量化公式（数据端 vs 查询端不对称）

查询和数据使用**不同的量化方式**，以适配 `dpbusd` 指令的 uint8 × int8 语义：

| 端 | 量化公式 | 存储类型 | 说明 |
|----|----------|----------|------|
| 数据端 | `int8_val = clamp(round(float_val + bias), -128, 127)` | int8 + float tail | dpbusd 第二操作数（有符号） |
| 查询端 | `uint8_val = clamp(round(float_val), 0, 255)` | uint8 | dpbusd 第一操作数（无符号） |

#### 4.3 向量存储布局

```
数据端：[ original_dim × int8 ] + [ 4 bytes float: sq_sum_half ]
         量化后的向量值              sum(float_val²) / 2.0

查询端：[ original_dim × uint8 ] + [ 4 bytes: 0 (unused padding) ]
         量化后的查询值              填零对齐，不参与计算
```

extra field 设计参考了 Record Quantizer 将额外信息附加在向量尾部的方式，但 SIFT Quantizer **只存储 sq_sum_half 一个字段**（4 字节），远小于 Record Quantizer 的 20 字节开销。

#### 4.4 距离计算的数学推导（search 路径）

设查询 q，数据 x（均为 float），则：

```
‖q - x‖² = ‖q‖² - 2⟨q, x⟩ + ‖x‖²
```

由于查询 q 对所有候选向量固定，‖q‖² 为常数，排序等价于：

```
dist = ‖x‖²/2 - ⟨q, x⟩ = sq_sum_half - ip(q, x)
```

其中 `ip(q, x)` 通过 `dpbusd(uint8_query, int8_data)` 高效计算。

> **注意**：dpbusd 计算的是 `Σ(q_uint8[i] * d_int8[i])`，其中
> `d_int8[i] = round(x[i]) + bias`，展开后为 `Σ(q[i] * round(x[i])) + bias·Σq[i]`。
> 由于 `bias·Σq[i]` 对固定查询是常数，不影响候选向量之间的排序，因此对搜索
> 结果正确性无影响。在标准 SIFT（global_min=0）场景下 bias=-128，此偏移量为
> `-128·Σq[i]`。

#### 4.5 Add 路径距离：对称 int8×int8 L2

**重要**：在 Vamana / HNSW 图的**构建阶段**，距离函数被用来比较**两个数据库
向量** a 和 b。此时两者都是**同一种量化方式**的 int8（带 bias），不能使用
dpbusd（会被误解释为 uint8×int8，语义错误导致 recall 急剧下降到 ~1%）。

因此 Metric 设计为双路径：

| 路径 | 使用场景 | 距离函数 | 语义 |
|------|----------|----------|------|
| add | 图构建（a,b 都是 int8 数据） | `uniform_squared_euclidean_int8_distance`（复用 uniform_int8 kernel） | `sum((a[i] - b[i])²)`，仅前 `original_dim` 字节，跳过 4 字节 tail |
| search | 查询（query=uint8, data=int8） | `sift_squared_euclidean_int8_distance` | `sq_sum_half - dpbusd(q, x)` |

两者都是真实 L2 的单调排序代理：
- add 路径：`(a_int8 - b_int8)² = (round(v_a) - round(v_b))² ≈ (v_a - v_b)²`
- search 路径：如上公式推导

实现上 `SiftInt8Metric::distance()` 返回 add 路径（`std::function` lambda
将 `dim` 减去 `sizeof(float)` 后转发到 uniform kernel），`SiftInt8Metric::
query_metric()` 返回 `SiftInt8QueryMetric` 实例，其 `distance()` 返回 sift
dpbusd 路径。Vamana/HNSW Streamer 在 `Open()` 时分别取 `add_distance_ =
metric_->distance()` 与 `search_distance_ = metric_->query_metric()
->distance()`。

#### 4.6 AVX-512 VNNI 距离计算（search 路径）

```
批量计算流程（每 batch_size=4 个向量为一组）：
1. 内积计算：
   对每 64 个元素：
     - 加载 uint8 query → zmm
     - 加载 int8 data[0..3] → zmm
     - dpbusd 累加：accs[i] += uint8 × int8（每4元素一组，累加到 int32）
     - 软件预取下一批数据
   标量尾部处理剩余元素
2. reduce：ip_results[i] = reduce_add(accs[i])  → float
3. 从每个向量尾部读取 sq_sum_half：
     tail = *(float*)(vector + original_dim)
4. 最终距离：distances[i] = sq_sum_half - ip_results[i]

剩余向量（不足一组）：逐个调用单向量距离函数，
  内部同样执行 dpbusd 内积 + 读取 sq_sum_half → 返回 sq_sum_half - ip。
```

> **与 MyQuant 的架构差异**：MyQuant 采用两阶段设计 —— 调用方先用
> `init_value`（sq_sum_half）初始化 distances 数组，批量内核中 `dists[i] -= ip`。
> 代码实现将两阶段合并为一步：内核自行读取 sq_sum_half 并直接赋值
> `distances[i] = sq_sum_half - ip`，使距离函数成为自包含接口，
> 与 zvec 框架的 Metric 接口契约一致。

#### 4.7 Normalize 行为

Reformer::normalize() 是 **no-op**（空操作）。距离值是单调排序代理，不是真实的 L2 距离，但排序顺序与真实 L2 完全一致。

---

## 三、两种 Quantizer 对比

| 特性 | Uniform Int8 | SIFT Int8 |
|------|-------------|-----------|
| Scale | 自动计算（可能为 1） | 固定为 1 |
| Bias | 自动计算 | 自动计算（更简化） |
| 查询/数据对称性 | 对称（都是 int8） | 不对称（query=uint8, data=int8） |
| Extra Field | 无 | sq_sum_half（4字节/向量） |
| 存储开销 | dim bytes | dim + 4 bytes |
| 距离含义 | 真实 L2（经 scale 还原） | 排序代理（单调等价） |
| SIMD 策略 | 差值平方 (vpmaddwd) | 内积 (dpbusd) |
| 每轮处理宽度 | 32 元素 | 64 元素 |
| 适用数据集 | 通用 | SIFT 类（非负整数，range ≤ 255） |
| 精度 | 有量化误差（scale≠1 时） | 无损（scale=1 且整数输入） |

---

## 四、源码清单

### Uniform Int8 Quantizer

| 文件 | 说明 |
|------|------|
| `src/core/quantizer/uniform_int8_converter.cc` | Converter 实现 |
| `src/core/quantizer/uniform_int8_reformer.cc` | Reformer 实现 |
| `src/core/metric/uniform_int8_metric.cc` | Metric 实现 |
| `src/turbo/avx512_vnni/uniform_int8/squared_euclidean.h` | SIMD 距离声明 |
| `src/turbo/avx512_vnni/uniform_int8/squared_euclidean.cc` | SIMD 距离实现 |

### SIFT Int8 Quantizer

| 文件 | 说明 |
|------|------|
| `src/core/quantizer/sift_int8_converter.cc` | Converter 实现 |
| `src/core/quantizer/sift_int8_reformer.cc` | Reformer 实现 |
| `src/core/metric/sift_int8_metric.cc` | Metric 实现（`SiftInt8` 对称 add 路径 + `SiftInt8Query` dpbusd search 路径） |
| `src/turbo/avx512_vnni/sift_int8/squared_euclidean.h` | SIMD 距离声明 |
| `src/turbo/avx512_vnni/sift_int8/squared_euclidean.cc` | SIMD 距离实现 |

### 共享修改

| 文件 | 修改内容 |
|------|----------|
| `src/include/zvec/turbo/turbo.h` | 添加 `QuantizeType::kSift` 枚举 |
| `src/turbo/turbo.cc` | 注册 sift_int8 距离函数 |
| `src/core/quantizer/quantizer_params.h` | 添加 `SIFT_INT8_REFORMER_BIAS` 常量 |
| `src/core/metric/metric_params.h` | 添加 `SIFT_INT8_METRIC_ORIGIN_METRIC_NAME` 常量 |
| `src/include/zvec/core/interface/index_param.h` | 添加 `QuantizerType::kSiftInt8` 枚举 |
| `src/core/interface/index.cc` | 添加 `kSiftInt8 → "SiftInt8StreamingConverter"` 分支 |

---

## 五、SIFT 1M 数据集验证结果

以 SIFT 1M（128 维、值域 [0, 218]）为基准，与旧的 Uniform Int8 方案对齐：

| 配置 | Recall@1 | Recall@10 | QPS |
|------|----------|-----------|-----|
| Vamana + UniformInt8 (ef=12) | 95.46% | 90.77% | 30,562 |
| Vamana + SiftInt8 (ef=12) | **95.13%** | 90.74% | **34,176** |
| HNSW + UniformInt8 (ef=29) | 94.92% | 90.66% | 30,432 |
| HNSW + SiftInt8 (ef=29) | 94.74% | 90.67% | **31,085** |
| HNSW + SiftInt8 (ef=80) | 99.15% | 98.22% | 13,759 |

在同等 recall 档位下，SIFT Int8 相比 Uniform Int8 获得 Vamana +12% / HNSW +2%
的 QPS 提升，验证 dpbusd 内积路径的性能优势。
