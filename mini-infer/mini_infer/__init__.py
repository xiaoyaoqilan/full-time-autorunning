"""mini-infer v0.1 — 单卡 LLM 推理引擎（教学级、可对标 vLLM/SGLang）。

路线图（README）：W1 调度器+HF backend baseline → W2 压测对标 → W3 分页/Radix KV+MLA
→ W4 MTP/EAGLE-3 投机解码 → W5 FP8/W4A16 量化 → W6 压测报告。
v0.1 已实现：分页 KV 管理（纯逻辑+引用计数前缀共享）、continuous batching +
chunked prefill 调度器、HF transformers 后端引擎、OpenAI 兼容服务、压测脚本。
"""
