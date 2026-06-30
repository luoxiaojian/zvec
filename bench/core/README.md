# Python Benchmark (sift / gist) — ann-benchmarks style

使用 zvec Python Collection API,按 [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
的调用方式完成 **fit → query → recall** 全流程,并报告 recall–QPS。索引超参与
仓库 `configs/`（及 `~/py/workspace/{sift,gist}/*/` 下的 `build.yaml` /
`search_po_pl.yaml`）保持一致。

唯一入口脚本：`ann_bench.py`。

## 与 ann-benchmarks 对齐的要点

- **fit（构建）**：`--build` 时按行顺序 `insert` + `optimize()`，使内部 doc id
  等于数据集行号（ann-benchmarks 依赖的契约）。
- **三段式查询计时**：`prepare_query` / `run_prepared_query` /
  `get_prepared_query_results`，只对 search 计时（排除入参准备），与榜单高分绑定
  （glass、kgn 等）一致。
- **返回整数行索引**：默认走 ``fast_query_doc_ids_only``；``--path ann_bench_doc_ids``
  走 ``ann_bench_prepare`` + ``ann_bench_search_doc_ids_only``（与 ann-benchmarks
  ``ZvecAnnBenchDocIds`` 相同）。
- **QPS = 1 / best_search_time**：在 `--runs` 次重复中取最优，口径同 ann-benchmarks。
- **Recall@k = |returned[:k] ∩ ground_truth[:k]| / k**，逐查询平均。

## 环境

```bash
cd bench/core
bash setup_venv.sh          # 创建 .venv 并安装 zvec（默认本地 editable）
source .venv/bin/activate
```

从 PyPI 安装 zvec：

```bash
ZVEC_INSTALL_MODE=pypi bash setup_venv.sh
```

## 数据目录

将训练/测试/ground-truth 放在同一目录，例如：

```
/data/sift/
  sift_train.vecs
  sift_test.txt
  sift_neighbors.txt
```

YAML 中的数据路径会被 `--data-dir` 下同名文件自动覆盖。

## 用法

### 一步完成：构建 + 压测 + 召回

```bash
python ann_bench.py --workspace-config ~/py/workspace/sift/hnsw_i8_m16 --build --force
```

### 仅压测/召回（collection 已存在）

```bash
python ann_bench.py --workspace-config ~/py/workspace/sift/hnsw_i8_m16
```

### recall–QPS 折中曲线（ann-benchmarks 标志输出，扫 ef）

```bash
python ann_bench.py --workspace-config ~/py/workspace/sift/hnsw_i8_m16 --ef 16,32,64,128,256
python ann_bench.py --workspace-config ~/py/workspace/sift/hnsw_i8_m16 --path ann_bench_doc_ids
```

`run_workspace_bench.py` 可与 `fast_query_doc_ids` / `fast_query` / `query` 并列跑
`ann_bench_doc_ids` 模式（默认 modes 已包含）：

```bash
python run_workspace_bench.py --dataset sift --modes ann_bench_doc_ids,fast_query_doc_ids
```

也可用 `--dataset {sift,gist}` 搭配 `--config <yaml>` / `--data-dir`，不依赖 workspace 目录。

常用参数：

- `--build` / `--force` / `--batch-size`：fit（构建）相关。
- `--runs N`：重复次数，QPS 取最优（默认 3）。
- `--count K`：topk（默认取配置中最大值）。
- `--ef a,b,c`：ef 扫描，每个 ef 输出一行 (recall, QPS)。
- `--path {fast_query_doc_ids,ann_bench_doc_ids}`：查询 API 链路（默认
  `fast_query_doc_ids`）。
- `--legacy-ids`：仅在 `--path fast_query_doc_ids` 时，用 `fast_query_doc_ids`
  （ids+scores）代替默认的 `fast_query_doc_ids_only`。

### （可选）文本转向量二进制

若只有 txt 训练集，可先转为 `.vecs`：

```bash
python txt2vecs.py --input train.txt --output sift_train.vecs --dimension 128
```

## 配置映射

**sift (HNSW)** — `build.yaml` / `search_po_pl.yaml`：L2、UNIFORM_INT8 量化、
`use_contiguous_memory=true`；查询 `ef_search`、`prefetch_*` 取自 yaml。

**gist (Vamana)** — L2、INT8 量化、`max_degree` / `search_list_size` / `alpha`
取自 yaml；查询 `ef_search`、`prefetch_*` 取自 yaml。
