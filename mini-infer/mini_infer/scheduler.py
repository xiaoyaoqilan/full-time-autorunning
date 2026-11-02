"""continuous batching + chunked prefill 调度器（纯逻辑，不依赖 torch，可单测）。

解决两个经典问题：
- head-of-line blocking：一个长 prompt 的 prefill 会卡住所有 decode 流。
  解法 = chunked prefill：每步只给 prefill 分配 token_budget 上限，其余留到后续步。
- 吞吐：decode 阶段把所有运行中序列拼成一个 batch 前向（continuous batching，
  序列完成即出队、新请求即时准入，不等整批）。

每步产出 StepPlan：
  prefills: [(seq_id, chunk_start, chunk_len)]   # 本步要跑的 prefill 块
  decodes : [seq_id]                             # 本步要 decode 的序列
公平策略：decode 优先保底（不被 prefill 挤占），剩余预算给 prefill（FCFS）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    seq_id: int
    prompt_len: int
    max_new_tokens: int
    prefill_done: int = 0          # 已完成 prefill 的 token 数（chunked 推进指针）
    decoded: int = 0               # 已 decode 的 token 数
    finished: bool = False

    @property
    def is_prefilling(self) -> bool:
        return self.prefill_done < self.prompt_len

    @property
    def is_done(self) -> bool:
        return self.decoded >= self.max_new_tokens


@dataclass
class StepPlan:
    prefills: list = field(default_factory=list)   # (seq_id, start, length)
    decodes: list = field(default_factory=list)    # seq_id 列表
    token_budget_used: int = 0


class ContinuousBatcher:
    def __init__(self, max_running: int = 32, token_budget: int = 2048):
        if token_budget < 1:
            raise ValueError("token_budget 须 >=1")
        self.max_running = max_running
        self.token_budget = token_budget
        self.waiting: deque = deque()
        self.running: dict[int, Request] = {}
        self._next_id = 0
        self._seq_counter = 0

    # ---- 请求生命周期 ----
    def submit(self, prompt_len: int, max_new_tokens: int) -> int:
        self._seq_counter += 1
        req = Request(seq_id=self._seq_counter, prompt_len=prompt_len, max_new_tokens=max_new_tokens)
        self.waiting.append(req)
        return req.seq_id

    def mark_prefill_done_chunk(self, seq_id: int, n_tokens: int):
        r = self.running[seq_id]
        r.prefill_done += n_tokens

    def mark_decoded(self, seq_id: int, n: int = 1):
        r = self.running[seq_id]
        r.decoded += n
        if r.is_done:
            r.finished = True

    def pop_finished(self) -> list[int]:
        out = [sid for sid, r in self.running.items() if r.finished]
        for sid in out:
            del self.running[sid]
        return out

    # ---- 每步调度 ----
    def plan(self) -> StepPlan:
        # 1) 准入：waiting → running（不超 max_running）
        while self.waiting and len(self.running) < self.max_running:
            r = self.waiting.popleft()
            self.running[r.seq_id] = r

        plan = StepPlan()
        budget = self.token_budget

        # 2) decode 保底：所有处于 decode 阶段的序列每步各 1 token
        decoding = [r for r in self.running.values() if not r.is_prefilling and not r.finished]
        plan.decodes = [r.seq_id for r in decoding]
        budget -= len(decoding)

        # 3) 剩余预算给 prefill（FCFS，chunked：每次最多 token_budget 内能给的量）
        for r in sorted(self.running.values(), key=lambda x: x.seq_id):
            if budget <= 0:
                break
            if not r.is_prefilling or r.finished:
                continue
            remain = r.prompt_len - r.prefill_done
            chunk = min(remain, budget)
            plan.prefills.append((r.seq_id, r.prefill_done, chunk))
            budget -= chunk
        plan.token_budget_used = self.token_budget - max(0, budget)
        return plan

    # ---- 统计 ----
    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "decoding": sum(1 for r in self.running.values() if not r.is_prefilling),
            "prefilling": sum(1 for r in self.running.values() if r.is_prefilling),
            "seq_counter": self._seq_counter,
        }
