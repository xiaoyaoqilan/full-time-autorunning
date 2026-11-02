# mini-infer 设计文档：量化模型与取舍

> 本文是项目的"深度底账"：每个模块的数学模型、参数依据、预期量级。所有数字分两类——【模型推算】（可复算）与【实测】（5090 上跑出来后回填）。两者不混写。

## 1. decode 吞吐的物理上限（为什么单流快不起来）

decode 每步必须把全部权重读一遍显存：

```
step_time ≈ weights_bytes / BW_eff + t_overhead
TPS_single ≈ 1 / step_time
```

RTX 5090：BW = 1792 GB/s，FP16 dense 算力远超 decode 所需 → **decode 是带宽问题，不是算力问题**。

以 Qwen3-8B（16.2GB bf16）为例【模型推算】：
- 有效带宽取 0.6~0.75（vLLM/SG 大 batch 实测区间）：step ≈ 16.2/(1792×0.675) ≈ 13.4ms
- 单流上限 ≈ **~75 tok/s**（不含 kernel 开销，加 2ms 开销 → ~65 tok/s）
- 这就是 W1 的"理论线"：实测定线方法 = 用 nvidia-smi dmon / torch profiler 拿真实 DRAM 吞吐，反推 bw_eff，误差预算 ±15%

推论：想快只有三条路——**减权重字节（量化）**、**把一次前向掰成多 token（投机解码）**、**提高并发摊薄权重读取（continuous batching）**。本项目的 W3-W5 就是这三条路。

## 2. batch 吞吐：为什么 continuous batching 有效

batch=B 时每步多读 B 份 KV，但权重只读一次：

```
step_time(B) = [W + B·seq·kv_per_token] / BW_eff + t_overhead
TPS_total(B) = B / step_time(B)
```

Qwen3-8B（GQA 36L/8H/128D，fp16 KV）【模型推算】：KV ≈ 2×36×8×128×2B = 147KB/token
- B=8、seq=4096：step ≈ (16.2+4.8)GB/1210GB/s ≈ 17.4ms → 整批 ≈ 460 tok/s
- B 越大越接近算力墙；批曲线上拐点 = 投机解码失效点（见 §4）

## 3. chunked prefill：队头阻塞的量化

无 chunked 时，一条 8K prompt 的 prefill ≈ 2×8e9×params_scale FLOPs，在 5090（~210 TFLOPS fp16 dense）上约 80-160ms，期间所有 decode 流停摆 → p99 尖刺。
限流 token_budget=C 后，单步 prefill 最长 C/8e9×… ；decode 等待上界从"整条 prompt"降到"C tokens"。
**实测项**：p99 ITL，无限流 vs C=1024/2048 两组。

## 4. 投机解码：α 模型与失效边界（本项目最值得做的实验）

一次验证接受 E[E] 个 token 的期望 ≈ (1-α^{k+1})/(1-α)；加速比 ≈ E × t_verify⁻¹ 相对单步。
- EAGLE-3 特征级 draft：α ≈ 0.8-0.9（文献），低并发 ITL 降 2-4×
- **失效边界（实测项，别人不报我们报）**：
  - batch ↑ → verify 从"带宽空闲"变"算力争抢"，加速比衰减；求临界 B*
  - constrained decoding（JSON grammar）→ draft token 被掩码否决，α 崩塌 → 期望加速 < 1
  - 温度采样 vs greedy、代码 vs 中文对话的 α 差异

## 5. 显存账本（5090 / 32GB）

| 项 | Qwen3-8B bf16 | Qwen3-8B W4A16 | Qwen3-14B int8 |
|---|---|---|---|
| 权重 | 16.2GB | 4.5GB | 14GB |
| 剩余给 KV（−15% 余量） | ~11GB | ~22.7GB | ~13.2GB |
| 128K 上下文 KV（fp16） | 19.1GB ✗ | ✅（需 KV 量化 fp8 → 9.6GB） | ✅ |

结论：**128K 长上下文在单卡上必须"权重量化 + KV 量化"二选一起步**——这也是 W5 的动机。

## 6. 与现有开源的差异化定位

| 项目 | 定位 | mini-infer 的差异 |
|---|---|---|
| vLLM / SGLang | 生产引擎，代码量 10 万+ | 我们是 1k 行级可读实现 + **失效边界实验叙事** |
| nano-vllm 等 mini 实现 | 精简复现 continuous batching | 我们加 2026 前沿三件套：MLA / MTP·EAGLE-3 / FP8·W4A16，且每个模块带"问题→方案→实测验证"闭环 |
| 教程文章 | 只讲原理 | 我们给出可复算的量化模型（本文）+ 实测回填的完整账本 |
