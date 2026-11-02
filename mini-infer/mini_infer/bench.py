"""压测：TTFT / ITL / 吞吐-并发曲线（5090 上运行，需 torch）。

用法:
  python -m mini_infer.bench --model Qwen/Qwen3-4B --concurrency 1 4 8 16 --n 32 -o bench_report.md
输出 markdown 报告，含与 vLLM/SGLang 同配置基线的对照表（基线列自己跑框架补）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time


def bench_one(eng, prompts, concurrency: int, max_new: int) -> dict:
    import threading
    lat_ttft, lat_e2e, tok_counts = [], [], []
    lock = threading.Lock()

    def worker(prompt):
        import torch
        ids = eng.tok(prompt, return_tensors="pt").input_ids[0].tolist()
        sid = eng.batcher.submit(len(ids), max_new)
        eng.kv.ensure_capacity(sid, len(ids) + max_new)
        eng.seqs[sid] = {"input_ids": ids, "output_ids": []}
        t0 = time.perf_counter()
        ttft = None
        while True:
            plan = eng.batcher.plan()
            if not plan.prefills and not plan.decodes:
                break
            chunk = {s: (a, b) for s, a, b in plan.prefills}
            for s, a, b in plan.prefills:
                eng._forward([s], chunk_infos=chunk)
                eng.batcher.mark_prefill_done_chunk(s, b)
                if ttft is None and s == sid:
                    with lock:
                        lat_ttft.append(time.perf_counter() - t0)
                    ttft = True
            for s in plan.decodes:
                eng._forward([s])
                eng.batcher.mark_decoded(s)
            for s in eng.batcher.pop_finished():
                eng.kv.free(s)
        dt = time.perf_counter() - t0
        with lock:
            lat_e2e.append(dt)
            tok_counts.append(len(eng.seqs[sid]["output_ids"]))

    threads = [threading.Thread(target=worker, args=(prompts[i % len(prompts)],))
               for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    total_tokens = sum(tok_counts)
    return {
        "concurrency": concurrency,
        "ttft_p50_ms": round(statistics.median(lat_ttft) * 1000, 1) if lat_ttft else None,
        "e2e_p50_s": round(statistics.median(lat_e2e), 2),
        "total_tokens": total_tokens,
        "wall_s": round(wall, 2),
        "throughput_tps": round(total_tokens / wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--n", type=int, default=32, help="提示词池大小")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("-o", default="bench_report.md")
    args = ap.parse_args()

    from .engine import Engine
    eng = Engine(args.model)
    prompts = [f"第 {i} 题：简要解释一个推理优化技术（KV cache / 投机解码 / 量化 / 前缀缓存 / chunked prefill）。"
               for i in range(args.n)]
    rows = [bench_one(eng, prompts, c, args.max_new) for c in args.concurrency]

    lines = [
        f"# mini-infer 压测报告（{args.model}）", "",
        f"- 硬件：RTX 5090 32GB ｜ max_new={args.max_new} ｜ 提示词池 {args.n} 条", "",
        "| 并发 | TTFT p50 (ms) | E2E p50 (s) | 吞吐 (tok/s) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['concurrency']} | {r['ttft_p50_ms']} | {r['e2e_p50_s']} | {r['throughput_tps']} |")
    lines += ["", "## 对照基线（同卡同模型自行补测）", "",
              "| 引擎 | 并发 | 吞吐 (tok/s) | 备注 |", "|---|---|---|---|",
              "| vLLM | - | - | 连续批处理基线 |", "| SGLang | - | - | RadixAttention 基线 |", ""]
    with open(args.o, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] 报告 → {args.o}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
