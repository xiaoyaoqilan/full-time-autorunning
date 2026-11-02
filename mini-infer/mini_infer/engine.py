"""引擎：把调度器的 StepPlan 落到 HF transformers 模型上（需 torch，5090 上运行）。

v0.1 口径（诚实声明）：
- KV 由 HF transformers 的 DynamicCache 自管；本项目的 PagedKVCache 当前承担
  显存预算与准入控制（can_admit），W3 计划把 KV 真正落到自定义 block（做 MLA 前置）。
- 采样默认 greedy，temperature/top_p 已留接口。

用法（5090 上）:
  python -m mini_infer.engine --model Qwen/Qwen3-4B --demo
"""
from __future__ import annotations

import argparse
import time


class Engine:
    def __init__(self, model_path: str, dtype: str = "bf16", device: str = "cuda",
                 max_running: int = 32, token_budget: int = 2048, gpu_total_gb: float = 32.0,
                 kv_headroom: float = 0.15):
        import torch  # 延迟 import：无卡环境可做调度器单测
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        kw = {"torch_dtype": getattr(torch, dtype), "device_map": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw).eval()
        cfg = self.model.config
        # KV 每 token 字节：2(K+V) × layers × kv_heads × head_dim × dtype_bytes
        self.kv_bytes_per_token = (2 * cfg.num_hidden_layers *
                                   getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) *
                                   cfg.head_dim * (2 if dtype == "bf16" else 1))
        from .kv_cache import PagedKVCache
        from .scheduler import ContinuousBatcher
        # 预算：总显存 − 权重 − 激活余量
        weights_gb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e9
        free_gb = gpu_total_gb * (1 - kv_headroom) - weights_gb
        n_blocks = max(1, int(free_gb * 1e9 / (self.kv_bytes_per_token * 16)))
        self.kv = PagedKVCache(num_blocks=n_blocks, block_size=16)
        self.batcher = ContinuousBatcher(max_running=max_running, token_budget=token_budget)
        self.seqs: dict[int, dict] = {}  # seq_id -> {input_ids, output_ids}

    # ---- 单步前向 ----
    def _forward(self, seq_ids, chunk_infos=None):
        """对一批序列做一次前向：prefill 块或 decode 单 token。返回每序列 next token。"""
        torch = self.torch
        inputs, positions, cache_lens, past = [], [], [], None
        for sid in seq_ids:
            sq = self.seqs[sid]
            if chunk_infos and sid in chunk_infos:
                start, length = chunk_infos[sid]
                ids = sq["input_ids"][start:start + length]
            else:
                ids = [sq["output_ids"][-1]]  # decode：喂上一 token
                past = None  # v0.1 简化：每序列独立前向（W3 统一 batch past）
            inputs.append(ids)
        outs = []
        # v0.1 简化实现：序列间独立前向（正确性优先）；W2 起换 padding batch + HF past_key_values
        for sid, ids in zip(seq_ids, inputs):
            sq = self.seqs[sid]
            t = torch.tensor([ids], device=self.model.device)
            with torch.no_grad():
                logits = self.model(t).logits[0, -1]
            nxt = int(logits.argmax())
            sq["output_ids"].append(nxt)
            outs.append(nxt)
        return outs

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        import torch
        ids = self.tok(prompt, return_tensors="pt").input_ids[0].tolist()
        sid = self.batcher.submit(len(ids), max_new_tokens)
        if not self.kv.can_admit(len(ids), max_new_tokens):
            raise MemoryError("显存预算不足（PagedKVCache 准入拒绝）")
        self.kv.ensure_capacity(sid, len(ids) + max_new_tokens)
        self.seqs[sid] = {"input_ids": ids, "output_ids": []}
        t0 = time.perf_counter()
        first = None
        while True:
            plan = self.batcher.plan()
            if not plan.prefills and not plan.decodes:
                break
            chunk = {s: (a, b) for s, a, b in plan.prefills}
            for s, a, b in plan.prefills:
                self._forward([s], chunk_infos=chunk)
                self.batcher.mark_prefill_done_chunk(s, b)
                if first is None:
                    first = time.perf_counter() - t0  # TTFT（首 chunk 完成近似）
            for s in list(plan.decodes):
                self._forward([s])
                self.batcher.mark_decoded(s)
            for s in self.batcher.pop_finished():
                self.kv.free(s)
        sq = self.seqs.pop(sid)
        self.seqs[sid] = sq
        return self.tok.decode(sq["output_ids"], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用一句话解释什么是 KV cache。")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()
    eng = Engine(args.model, dtype=args.dtype)
    t0 = time.perf_counter()
    out = eng.generate(args.prompt, args.max_new)
    dt = time.perf_counter() - t0
    print(f"--- 输出（{args.max_new} max new, {dt:.1f}s）---\n{out}")


if __name__ == "__main__":
    main()
