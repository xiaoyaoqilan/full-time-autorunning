# nano-vllm 实验室：5090 运行手册 + 实验计划

> 上游：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（MIT，~1.4k 行，15.5k★）
> 本目录 = 仓库内 vendored 副本（`../nano-vllm/`，保留原 LICENSE 与署名）+ 我们的运行手册、源码精读、实验计划。
> 定位转变：不自研轮子，**站在最值得抄的开源引擎上做复现 → 对标 → 增量实验**。

## 一、5090 运行手册（逐条可执行）

```bash
# 0. 环境（5090 = Blackwell SM120，装最新 CUDA 版 torch）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git   # 或用本仓库 vendored 副本: pip install -e ./nano-vllm
pip install modelscope   # 国内下载权重备用

# 1. 权重（0.6B 跑通 → 4B/8B 上强度）
modelscope download --model Qwen/Qwen3-0.6B --local_dir ~/huggingface/Qwen3-0.6B
# modelscope download --model Qwen/Qwen3-4B --local_dir ~/huggingface/Qwen3-4B

# 2. 冒烟：跑通 example
python ../nano-vllm/example.py

# 3. 复现官方基准（256 序列，100-1024 随机长度）
python ../nano-vllm/bench.py     # 改 path 为本机模型路径

# 4. 对标 vLLM（同一负载）
pip install vllm
# 注释切换 bench.py 里的 import，跑出 vLLM 数字 → 填入下方基准表
```

## 二、基准表（跑完回填）

| 引擎 | 模型 | 硬件 | 总 token | 耗时 | 吞吐 |
|---|---|---|---|---|---|
| nano-vllm（官方） | Qwen3-0.6B | 4070 Laptop 8GB | 133,966 | 93.41s | 1434.13 |
| vLLM（官方同负载） | Qwen3-0.6B | 4070 Laptop 8GB | 133,966 | 98.37s | 1361.84 |
| nano-vllm | Qwen3-0.6B | **5090 32GB** | - | - | - |
| vLLM | Qwen3-0.6B | **5090 32GB** | - | - | - |
| nano-vllm | Qwen3-4B | **5090 32GB** | - | - | - |
| vLLM | Qwen3-4B | **5090 32GB** | - | - | - |

**预期（模型推算）**：5090 带宽 1792GB/s，0.6B（1.2GB bf16）理论单流 ~1200 tok/s；4B（8GB）单流 ~170 tok/s、batch 吞吐看 KV 预算。实测偏离理论 >25% 就去查 kernel/bw_eff。

## 三、源码精读路线（1200 行怎么读）

按数据流读，每块对应一次 commit 笔记（`notes/` 按编号追加）：

1. `engine/llm_engine.py` — generate/step 主循环：请求怎么变成每步的 batch
2. `engine/scheduler.py` — 调度：prefill/decode 混批、抢占、准入
3. `engine/block_manager.py` — 分页 KV + 前缀缓存（xxhash 内容寻址块）
4. `engine/model_runner.py` — CUDA Graph 捕获/回放、TP 进程间通信
5. `models/qwen3.py` + `layers/` — 模型与算子：FlashAttention 变长批处理、RMSNorm、RoPE、并行线性层
6. `bench.py` — 基准方法学（为什么 ignore_eos、固定 seed、warmup）

## 四、增量实验（精读完再动，每个都是简历素材）

| 实验 | 内容 | 产出 |
|---|---|---|
| E1 复现 | 官方基准在 5090 重跑 + vLLM 对照 | 上表回填 + 偏差分析（bw_eff 反推） |
| E2 长上下文 | 0.6B/4B 撑 32K/64K 上下文，测 KV 压力与吞吐衰减 | 5090 32GB 的长度-并发包络线 |
| E3 投机解码 | 接 EAGLE-3 / n-gram prompt lookup，测 α 与加速比 | 含高并发失效边界的完整曲线 |
| E4 量化 | W8A8/W4A16 权重替换，精度回退（MMLU/GSM8K） | 显存-精度-速度三角实测 |
| E5 serving | 上游 Rust OpenAI 前端压测（bench serve harness） | 在线 vs 离线吞吐差与瓶颈定位 |

## 五、诚实声明

- 上游基准（1434 vs 1362 tok/s）是 0.6B 小模型 + 单卡离线批推理场景，**不代表 nano-vllm 优于 vLLM**——它赢在没有生产框架的开销；大模型/高并发/多卡下 vLLM 工程深度才体现。面试里把这个说清楚反而是加分项。
- 沙箱（本开发环境）无 CUDA，本文所有"实测"数字需在 5090 机器回填；回填前不接受任何未实测数字写进简历。
