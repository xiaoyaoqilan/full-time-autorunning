"""分页 KV cache 管理（纯逻辑，不依赖 torch，可单测）。

设计对齐 vLLM PagedAttention 的核心思想：
- 物理显存切成固定大小 block（每 block 存 block_size 个 token 的 K/V）
- 每个序列持有 block_table（逻辑块号 → 物理块号）
- block 引用计数支持前缀共享（copy-on-write）：多个序列共享同一前缀块，
  某序列要写共享块时先复制再写

v0.1 状态：作为显存预算/准入控制的真实数据结构；W3 计划把 HF 的 KV 真正
落到自定义 block 上（当前 HF forward 内部自管 KV，本结构负责预算与共享统计）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    id: int
    ref_count: int = 0
    seq_ids: set = field(default_factory=set)  # 引用该块的序列（前缀共享诊断用）


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int = 16):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks/block_size 须 >=1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: dict[int, Block] = {i: Block(id=i) for i in range(num_blocks)}
        self.free_list: list[int] = list(range(num_blocks - 1, -1, -1))  # LIFO，热块友好
        self.block_tables: dict[int, list[int]] = {}  # seq_id -> 物理块号列表
        self.seq_lens: dict[int, int] = {}            # seq_id -> 真实 token 数

    # ---- 容量 ----
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_list)

    def fragmentation(self) -> float:
        """块内碎片率：已分配块中未被 token 占满的比例。"""
        used = self.used_blocks()
        if used == 0:
            return 0.0
        waste = 0
        for table in self.block_tables.values():
            for bid in table:
                n_share = self.blocks[bid].ref_count
                waste += self.block_size - 1  # v0.1 粗口径：非满块计满碎（精确口径 W3 接 token 计数）
        return min(1.0, waste / max(1, used * self.block_size))

    # ---- 分配 ----
    def _alloc_block(self, seq_id: int) -> int:
        if not self.free_list:
            raise MemoryError(f"KV cache 满：序列 {seq_id} 申请 block 失败（{self.num_blocks} blocks 全占用）")
        bid = self.free_list.pop()
        self.blocks[bid].ref_count = 1
        self.blocks[bid].seq_ids = {seq_id}
        return bid

    def ensure_capacity(self, seq_id: int, n_tokens: int) -> int:
        """确保序列可容纳 n_tokens 个 token，返回该序列当前 block 数。"""
        table = self.block_tables.setdefault(seq_id, [])
        need = (n_tokens + self.block_size - 1) // self.block_size
        while len(table) < need:
            table.append(self._alloc_block(seq_id))
        return len(table)

    def append_slot(self, seq_id: int) -> tuple[int, int]:
        """decode 追加 1 token 的槽位，返回 (block_id, slot_idx)。"""
        n = self.tokens_of(seq_id) + 1
        self.ensure_capacity(seq_id, n)
        self.seq_lens[seq_id] = n
        table = self.block_tables[seq_id]
        bid = table[(n - 1) // self.block_size]
        return bid, (n - 1) % self.block_size

    def tokens_of(self, seq_id: int) -> int:
        return self.seq_lens.get(seq_id, 0)

    # ---- 释放与共享 ----
    def free(self, seq_id: int):
        self.seq_lens.pop(seq_id, None)
        for bid in self.block_tables.pop(seq_id, []):
            blk = self.blocks[bid]
            blk.ref_count -= 1
            blk.seq_ids.discard(seq_id)
            if blk.ref_count <= 0:
                blk.seq_ids = set()
                self.free_list.append(bid)

    def fork_prefix(self, src_seq: int, dst_seq: int, n_tokens: int):
        """前缀共享：dst 引用 src 前 n_tokens 对应的块（copy-on-write 语义由写方负责）。"""
        if n_tokens % self.block_size != 0:
            raise ValueError("fork 长度须按 block 对齐")
        src_table = self.block_tables.get(src_seq, [])
        n_blocks = n_tokens // self.block_size
        if n_blocks > len(src_table):
            raise ValueError("fork 长度超过源序列")
        self.free(dst_seq)  # 若 dst 已有表则先释放
        table = []
        for bid in src_table[:n_blocks]:
            blk = self.blocks[bid]
            blk.ref_count += 1
            blk.seq_ids.add(dst_seq)
            table.append(bid)
        self.block_tables[dst_seq] = table
        self.seq_lens[dst_seq] = n_tokens

    def shared_block_ratio(self) -> float:
        used = self.used_blocks()
        if used == 0:
            return 0.0
        shared = sum(1 for b in self.blocks.values() if b.ref_count > 1)
        return shared / used

    # ---- 准入控制 ----
    def can_admit(self, prompt_tokens: int, max_new_tokens: int) -> bool:
        need = (prompt_tokens + max_new_tokens + self.block_size - 1) // self.block_size
        return len(self.free_list) >= max(0, need)
