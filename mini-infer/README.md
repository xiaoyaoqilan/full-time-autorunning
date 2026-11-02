# mini-infer

单卡 LLM 推理引擎（教学级）：在一张 RTX 5090（32GB）上从零实现 2026 主流 serving 栈的核心技术，全部性能对标 vLLM/SGLang 同配置实测。

> 为什么自己写： continuous batching、chunked prefill、分页 KV、前缀缓存、投机解码、量化——这些是现代推理框架的全部骨架，也是面试考点。跑通一遍并拿到对标数据，比只用框架懂得多得多。

## 设计思路：五个问题，五条验证

每一个模块都从真实问题出发，每条方案都有对应的可测验证项——这也是整个项目的叙事主线：

| # | 问题 | 方案 | 验证（实测项） |
|---|------|------|---------------|
| 1 | decode 慢：每生成一个 token 都要读一遍全部权重，显存带宽主导 | 分页 KV 管理（PagedAttention 式）+ continuous batching 把空闲算力换成并发 | 吞吐-并发曲线逼近 1792GB/s 理论线；对标 vLLM 偏差 |
| 2 | 长 prompt 的 prefill 卡住所有 decode 流（队头阻塞），p99 尾延迟尖刺 | chunked prefill：每步 prefill 限流，decode 保底 | 同负载下 p99 对比无调度限流版本 |
| 3 | 多轮对话/RAG 反复重算相同前缀 | RadixAttention 式前缀缓存 + block 引用计数（copy-on-write） | 前缀复用负载下吞吐提升倍数 |
| 4 | 每 token 一次前向太浪费 | MTP / EAGLE-3 投机解码：一次前向验证多个 draft token | α 接受率与加速比；**高并发/约束解码下的失效边界**（这是已知的坑，测出来比只报加速比更值钱） |
| 5 | 32GB 放不下大模型长上下文 | FP8 / W4A16 量化 | MMLU/GSM8K 精度回退 < 2%；显存-精度-速度三角 |

## v0.1 已实现

| 模块 | 状态 | 说明 |
|---|---|---|
| 分页 KV cache（`kv_cache.py`） | ✅ 纯逻辑+单测 | block 分配/释放、引用计数、fork 前缀共享（copy-on-write 语义）、准入控制；纯标准库，无 GPU 可测 |
| 调度器（`scheduler.py`） | ✅ 纯逻辑+单测 | continuous batching + chunked prefill（decode 保底、prefill 限流，治队头阻塞） |
| 引擎（`engine.py`） | ✅ 需 GPU | HF transformers 后端 + 上述调度/KV 预算；KV 真落自定义 block 是 W3 |
| 压测（`bench.py`） | ✅ 需 GPU | TTFT / E2E / 吞吐-并发曲线，输出 markdown 报告（含 vLLM/SGLang 对照空表） |

## 快速开始

```bash
# 无 GPU 环境（逻辑层验证）
python tests/test_core.py

# 5090 上真跑
pip install torch transformers accelerate
python -m mini_infer.engine --model Qwen/Qwen3-4B --prompt "用一句话解释 KV cache"
python -m mini_infer.bench --model Qwen/Qwen3-4B --concurrency 1 4 8 16 -o bench_report.md
```

## 路线图

| 周 | 里程碑 | 验收 |
|---|---|---|
| W1 | 调度器 + HF backend baseline（本版） | 单测 6/6；单流 decode 跑通 |
| W2 | padding batch + HF past_key_values 真批处理；压测对标 vLLM | 吞吐-并发曲线 vs vLLM 偏差 < 30% |
| W3 | KV 真落自定义 block（替换 DynamicCache）+ RadixAttention 前缀复用 | 前缀共享负载吞吐提升可测 |
| W4 | 投机解码：先接 EAGLE-3 权重，再自训 draft head | 报告 α 接受率与加速比；含高并发失效分析 |
| W5 | FP8/W4A16 量化路径 | MMLU/GSM8K 回退 < 2% |
| W6 | 完整压测报告 + 调优复盘 | decode 带宽利用率 vs 1792GB/s 理论线 |

## 诚实声明

- 所有系数/性能数字以实测为准，README 与报告中不预填任何未实测数据。
- v0.1 的 KV cache 是显存预算与共享逻辑的真实实现，但 HF forward 内部 KV 仍由 transformers 自管——W3 替换。
- 单序列独立前向是 v0.1 的正确性优先实现，真批处理在 W2。

## License

MIT
