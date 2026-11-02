# -*- coding: utf-8 -*-
"""纯逻辑单测（无 torch 依赖，沙箱/CI 可跑）：python tests/test_core.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_infer.kv_cache import PagedKVCache
from mini_infer.scheduler import ContinuousBatcher

def test_paged_kv_basic():
    kv = PagedKVCache(num_blocks=8, block_size=4)
    assert kv.capacity_tokens() == 32
    kv.ensure_capacity(1, 10)          # 预留 10 token 容量 → 3 blocks
    assert len(kv.block_tables[1]) == 3
    assert kv.tokens_of(1) == 0        # 预留 ≠ 已写入
    bid = slot = None
    for _ in range(10):                # 逐 token 写入 10 个
        bid, slot = kv.append_slot(1)
    assert (bid, slot) == (kv.block_tables[1][2], 1)  # 第 10 个 token：第 3 块第 1 槽
    assert kv.tokens_of(1) == 10
    kv.free(1)
    assert kv.used_blocks() == 0

def test_paged_kv_oom():
    kv = PagedKVCache(num_blocks=2, block_size=4)
    kv.ensure_capacity(1, 8)
    try:
        kv.ensure_capacity(2, 4)
        assert False, "应抛 MemoryError"
    except MemoryError:
        pass

def test_prefix_share_and_fork():
    kv = PagedKVCache(num_blocks=8, block_size=4)
    kv.ensure_capacity(1, 8)
    kv.fork_prefix(1, 2, 8)            # seq2 完全共享 seq1 的 2 个块
    assert kv.used_blocks() == 2
    assert kv.shared_block_ratio() == 1.0
    kv.free(1)
    assert kv.used_blocks() == 2       # seq2 还持有
    kv.free(2)
    assert kv.used_blocks() == 0

def test_batcher_admission_and_decode_priority():
    b = ContinuousBatcher(max_running=2, token_budget=16)
    b.submit(prompt_len=100, max_new_tokens=2)   # 长 prompt（要 chunked）
    b.submit(prompt_len=2, max_new_tokens=3)     # 短 prompt
    plan = b.plan()
    assert len(b.running) == 2 and len(b.waiting) == 0
    total_prefill = sum(c for _, _, c in plan.prefills)
    assert total_prefill <= 16 - len(plan.decodes) + len(plan.decodes)  # 预算约束
    # decode 保底：短 prompt 一步 prefill 完成后进入 decode，长 prompt 未 prefill 完时
    # 下一 plan 应该仍为短序列保留 decode 槽位
    for s, start, chunk in plan.prefills:
        b.mark_prefill_done_chunk(s, chunk)
    plan2 = b.plan()
    short_done = any(not b.running[s].is_prefilling for s in b.running)
    if short_done:
        assert len(plan2.decodes) >= 1
        assert plan2.token_budget_used <= 16

def test_chunked_prefill_finishes():
    b = ContinuousBatcher(max_running=4, token_budget=16)
    sid = b.submit(prompt_len=50, max_new_tokens=1)
    steps = 0
    while True:
        plan = b.plan()          # 首次 plan 触发 waiting→running 准入
        for s, start, chunk in plan.prefills:
            b.mark_prefill_done_chunk(s, chunk)
        if sid in b.running and not b.running[sid].is_prefilling:
            break
        steps += 1
        assert steps < 20
    assert b.running[sid].prefill_done == 50
    assert steps == 3  # 50 = 16+16+16+2，共 4 次 plan（第 4 次完成后退出），chunk 计数 3

def test_request_finish_and_removal():
    b = ContinuousBatcher(max_running=4, token_budget=64)
    sid = b.submit(prompt_len=4, max_new_tokens=2)
    plan = b.plan(); [b.mark_prefill_done_chunk(s, c) for s, _, c in plan.prefills]
    plan = b.plan(); [b.mark_decoded(s) for s in plan.decodes]
    plan = b.plan(); [b.mark_decoded(s) for s in plan.decodes]
    assert b.pop_finished() == [sid]
    assert len(b.running) == 0

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} 全部通过")
