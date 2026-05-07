# Int8 Quantizer 设计文档

## 一、概述

Uniform Int8 Quantizer 是 zvec 对外暴露的唯一 int8 量化接口
（`QuantizerType::kUniformInt8`）。它采用**全局统一的 scale/bias**
将 float32 向量量化为 int8 向量，使得 L2 距离可以完全在整数域中计算，
无需逐向量反量化。

为了在特定数据分布上获得更高性能，Converter 内部在训练阶段会自动检测
数据特征，并在满足条件时切换到一条**非负整数 fast-path**（scale=1、
采用 dpbusd 内积的不对称距离）。整个切换过程对用户完全透明：

- 用户只选 `kUniformInt8` 一个类型
- 训练时根据数据自动分派到 **通用路径（symmetric L2）** 或
  **非负整数 fast-path（dpbusd 内积）**
- 持久化 meta 中记录实际使用的 reformer / metric 名字；搜索端
  `Index::Open` 根据磁盘 meta 自动重建对应组件，不需要用户干预

---

## 二、使用方式

| 组件 | 对外注册名 | 说明 |
|------|-----------|------|
| Converter | `UniformInt8StreamingConverter` | 训练阶段计算参数，自动选择内部路径 |
| Reformer（通用路径） | `UniformInt8StreamingReformer` | 对称 float→int8 量化 |
| Metric（通用路径） | `UniformInt8` | 对称 int8×int8 L2 距离 |

用户只需在 `quantizer_param.type` 中填写 `kUniformInt8` 即可，不暴露
fast-path 的开关或类型。

**关键参数（自动计算，无需手动设置）：**

- `uniform_int8.reformer.scale` — 全局缩放因子（fast-path 下固定为 1）
- `uniform_int8.reformer.bias` — 全局偏移量
- `proxima.uniform_int8.metric.origin_metric_name` — 原始距离类型
  （目前仅支持 `SquaredEuclidean`）

**Turbo 层枚举：** `turbo::QuantizeType::kUniform` /
`turbo::QuantizeType::kUnitScale`（内部使用，不对外暴露）。

### 适用范围

| 维度 | 支持范围 |
|------|----------|
| 数据集 | 任意 float32 向量数据集（值域不限） |
| 量化目标类型 | `DT_INT8` |
| 距离类型 | `SquaredEuclidean` |
| 硬件要求 | AVX-512 VNNI（无 VNNI 时回退到标量路径） |

---

## 三、实现思路

### 3.1 训练流程（Converter::train）

1. 扫描全部训练向量，统计 `global_min` / `global_max`，并检查是否全部为
   整数（`all_integer`）
2. **Fast-path 检测**：若 `all_integer && global_min ≥ 0 && global_max ≤ 255`
   （即数据为非负整数且落在 [0, 255] 内），则委托给内部实现
   `UnitScaleInt8StreamingConverter`，后续所有 `transform / dump / result / meta`
   调用都转发给该 delegate，同时改写 meta 中的 reformer / metric /
   converter 名字为 fast-path 对应的内部名称
3. **通用路径**：若未满足 fast-path 条件，则按 Uniform 方式执行：
   - 全局整数可无损路径：若所有值为整数且 range ≤ 254 → `scale = 1`
   - 否则线性缩放：`scale = 254 / range`，`bias = -global_min × scale - 127`
4. 将 scale/bias 写入 reformer_params 和 converter_params 持久化

内部 delegate 通过 `IndexFactory::CreateConverter(...)` 创建；若 delegate
创建或训练失败会自动退化到通用路径，保证健壮性。

### 3.2 通用路径（Symmetric Int8 L2）

#### 量化公式

查询和数据端采用**完全相同**的量化公式：

```
正向：int8_val = clamp(round(float_val × scale + bias), -127, 127)
反向：float_val ≈ (int8_val - bias) / scale
```

#### 距离计算（AVX-512）

```
对每 32 个元素：
  1. 加载 int8 → 符号扩展 int16 (vpmovsx)
  2. 差值 diff = query - vec (vpsubw)
  3. 差值平方 + 邻对求和 (vpmaddwd) → int32
  4. 累加 (vpaddd)
最终 reduce 为单个 int32 → cast to float
```

距离关系：`int8_L2 = scale² × real_L2`，通过 Reformer 的 `normalize()`
乘以 `1/scale²` 还原。

#### 向量存储布局

```
数据端：[ dim × int8 ]
查询端：[ dim × int8 ]（与数据端对称，无 extra fields）
```

---

### 3.3 非负整数 Fast-path（Scale=1, dpbusd 内积）

此路径针对 **scale=1 可无损量化**的场景设计：当训练数据全部为非负
整数且值域落在 [0, 255] 时，可以直接以恒等缩放（scale=1）将 float32
映射为 int8/uint8，避免任何量化误差。典型样例包括 SIFT、GIST 等图像
特征向量，但该路径并不与具体数据集强绑定，任何满足上述条件的数据
分布都可自动命中。

**核心思想**：将每个向量的**平方和的一半**（`sq_sum / 2.0f`）作为尾部
附加字段存储，使距离计算可以利用 **dpbusd 指令**（uint8 × int8 内积）
替代差值平方求和，获得显著性能优势。

#### 训练

1. 遍历全部训练向量，统计 `global_min`
2. 计算 `bias = -round(global_min) - 128`（将 global_min 映射到 int8 的 -128）
3. 将 bias 写入 reformer_params 持久化

**无需计算 scale**（固定为 1），训练过程更简洁。

#### 量化公式（不对称）

查询和数据使用**不同的量化方式**以适配 `dpbusd` 指令的 uint8 × int8 语义：

| 端 | 量化公式 | 存储类型 |
|----|----------|----------|
| 数据端 | `int8_val = clamp(round(float_val + bias), -128, 127)` | int8 + float tail |
| 查询端 | `uint8_val = clamp(round(float_val), 0, 255)` | uint8 |

#### 向量存储布局

```
数据端：[ original_dim × int8 ] + [ 4 bytes float: sq_sum_half ]
         量化后的向量值              sum(float_val²) / 2.0

查询端：[ original_dim × uint8 ] + [ 4 bytes: 0 (unused padding) ]
         量化后的查询值              填零对齐，不参与计算
```

#### 距离计算的数学推导（search 路径）

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
> `d_int8[i] = round(x[i]) + bias`，展开后含常数偏移 `bias·Σq[i]`；
> 对固定查询而言该偏移是常数，不影响候选之间的排序。

#### Add 路径距离：对称 int8×int8 L2

**重要**：在 Vamana / HNSW 图的**构建阶段**，距离函数被用来比较**两个
数据库向量** a 和 b。此时两者都是**同一种量化方式**的 int8（带 bias），
不能使用 dpbusd（会被误解释为 uint8×int8，语义错误导致 recall 急剧
下降到 ~1%）。

因此 fast-path Metric 设计为双路径：

| 路径 | 使用场景 | 距离函数 | 语义 |
|------|----------|----------|------|
| add | 图构建（a,b 都是 int8 数据） | 复用 uniform_int8 kernel（symmetric L2） | `sum((a[i] - b[i])²)`，仅前 `original_dim` 字节，跳过 4 字节 tail |
| search | 查询（query=uint8, data=int8） | `unit_scale_squared_euclidean_int8_distance` | `sq_sum_half - dpbusd(q, x)` |

两者都是真实 L2 的单调排序代理，排序结果与真实 L2 完全一致。

内部实现上 `UnitScaleInt8Metric::distance()` 返回 add 路径，
`UnitScaleInt8Metric::query_metric()` 返回 `UnitScaleInt8QueryMetric` 实例负责
search 路径。Vamana / HNSW Streamer 在 `Open()` 时分别取
`add_distance_` 与 `search_distance_`。

#### AVX-512 VNNI 距离计算（search 路径）

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

剩余向量（不足一组）：逐个调用单向量距离函数。
```

#### Normalize 行为

fast-path Reformer 的 `normalize()` 是 **no-op**。距离值是单调排序代理，
不是真实的 L2 距离，但排序顺序与真实 L2 完全一致。

---

### 3.4 搜索端自动适配（Index::Open）

由于 fast-path 切换发生在训练阶段，持久化 meta 中的 reformer / metric
/ 维度可能与 `Index::Init()` 初始化的不一致。`Index::Open()` 在打开
storage 后、streamer 初始化前会执行如下逻辑：

1. 调用 `IndexHelper::DeserializeFromStorage` peek 磁盘上的 IndexMeta
2. 比较 `reformer_name / metric_name / dimension` 是否与当前 Index
   对象上的一致
3. 若不一致，则按磁盘 meta **重建 reformer / metric / streamer**，
   并更新 `streamer_vector_meta_`

这使得用户只要持续使用 `kUniformInt8`，构建与搜索侧的 fast-path 切换
就自动一致，无需任何额外配置。

---

## 四、两条路径对比（内部实现视角）

| 特性 | 通用路径 | 非负整数 fast-path |
|------|----------|--------------------|
| Scale | 自动计算（可能为 1） | 固定为 1 |
| Bias | 自动计算 | 自动计算（更简化） |
| 查询/数据对称性 | 对称（都是 int8） | 不对称（query=uint8, data=int8） |
| Extra Field | 无 | sq_sum_half（4字节/向量） |
| 存储开销 | dim bytes | dim + 4 bytes |
| 距离含义 | 真实 L2（经 scale 还原） | 排序代理（单调等价） |
| SIMD 策略 | 差值平方 (vpmaddwd) | 内积 (dpbusd) |
| 每轮处理宽度 | 32 元素 | 64 元素 |
| 适用数据 | 通用 | 非负整数、值域 ≤ 255 |
| 精度 | 有量化误差（scale≠1 时） | 无损（scale=1 且整数输入） |

---

## 五、源码清单

### 对外入口

| 文件 | 说明 |
|------|------|
| `src/include/zvec/core/interface/index_param.h` | `QuantizerType::kUniformInt8` 枚举 |
| `src/core/interface/index.cc` | `kUniformInt8 → "UniformInt8StreamingConverter"` 分派；`Index::Open()` 中基于磁盘 meta 的 reformer/metric/streamer 自动重建逻辑 |

### 通用路径（symmetric uniform int8）

| 文件 | 说明 |
|------|------|
| `src/core/quantizer/uniform_int8_converter.cc` | Converter 实现；包含 fast-path 检测与委托逻辑 |
| `src/core/quantizer/uniform_int8_reformer.cc` | Reformer 实现 |
| `src/core/metric/uniform_int8_metric.cc` | Metric 实现 |
| `src/turbo/avx512_vnni/uniform_int8/squared_euclidean.{h,cc}` | SIMD 距离实现 |

### 非负整数 fast-path（内部实现，未对外暴露为独立 QuantizerType）

| 文件 | 说明 |
|------|------|
| `src/core/quantizer/unit_scale_int8_converter.cc` | fast-path Converter；通过 factory 名字 `UnitScaleInt8StreamingConverter` 被内部创建 |
| `src/core/quantizer/unit_scale_int8_reformer.cc` | fast-path Reformer（uint8 query + int8 + tail） |
| `src/core/metric/unit_scale_int8_metric.cc` | fast-path Metric（add = symmetric L2；search = dpbusd） |
| `src/turbo/avx512_vnni/unit_scale_int8/squared_euclidean.{h,cc}` | fast-path SIMD 距离实现 |

### 共享组件

| 文件 | 说明 |
|------|------|
| `src/include/zvec/turbo/turbo.h` | `QuantizeType::kUniform` / `kUnitScale` 内部枚举 |
| `src/turbo/turbo.cc` | 注册 uniform / unit-scale int8 距离函数 |
| `src/core/quantizer/quantizer_params.h` | `*_REFORMER_BIAS` / `*_REFORMER_SCALE` 常量 |
| `src/core/metric/metric_params.h` | `*_METRIC_ORIGIN_METRIC_NAME` 常量 |

---

## 六、SIFT 1M 数据集验证结果

以 SIFT 1M（128 维、值域 [0, 218]，满足 scale=1 fast-path 触发条件）
为基准，对比 fast-path 自动切换前后：

| 配置 | Recall@1 | Recall@10 | QPS |
|------|----------|-----------|-----|
| Vamana + 通用路径 (ef=12) | 95.46% | 90.77% | 30,562 |
| Vamana + fast-path (ef=12) | **95.34%** | **90.81%** | **34,176** |
| HNSW + 通用路径 (ef=29) | 94.92% | 90.66% | 30,432 |
| HNSW + fast-path (ef=29) | 94.74% | 90.67% | **31,085** |
| HNSW + fast-path (ef=80) | 99.15% | 98.22% | 13,759 |

在同等 recall 档位下，fast-path 相比通用路径获得 Vamana +12% / HNSW +2%
的 QPS 提升，验证 dpbusd 内积路径的性能优势。对用户而言这只是
`kUniformInt8` 的一次内部优化，无需任何配置变更。
