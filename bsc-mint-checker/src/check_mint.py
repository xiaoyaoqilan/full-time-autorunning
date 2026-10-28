#!/usr/bin/env python3
"""check_mint.py — BSC 免费 mint 合约批量体检 CLI（单文件、仅 Python 标准库）。

用法:
    python check_mint.py contracts.txt -o report.md
    python check_mint.py contracts.txt            # 不加 -o 时报告输出到 stdout

功能:
    读取「名称,地址」格式的合约清单，通过 BSC 公共 JSON-RPC 多节点 failover
    逐合约执行检查管线，输出 markdown 体检报告（汇总表 + 逐合约详情）。

检查管线（每合约顺序执行）:
    1. check_existence      : eth_getCode 判定 合约 / EOA或已自毁（确定性）
    2. check_interface      : ERC-165 supportsInterface 识别 ERC-721/1155（确定性）
    3. check_fee_heuristics : eth_call 常量探测 price/mintPrice 等（启发式）
    4. check_limit_heuristics: eth_call 探测 maxSupply/maxMintAmount（启发式）
    5. check_bytecode_flags : 字节码特征匹配 strong/weak 两级（启发式）
    6. bscscan 可选项       : 配置 BSCSCAN_API_KEY 时做开源验证；无 key 时
                              结论为「未验证，风险未知，按 ⚠️ 计入建议」

遵守 04-算法设计「dev 必守不变式」:
    (1) max_workers=4、RPC timeout=10s、退避 0.5s→1s→2s、单合约 60s 预算，
        其中并发数可经 --workers 覆盖（默认 4 不改），其余为代码常量；
    (2) JSON-RPC error 属业务层错误，不换节点重试；仅网络类错误（超时/429/5xx/
        连接错误）走节点 failover + 指数退避（含 ±20% 抖动）；例外：
        节点级鉴权失败（auth 类 error，如 Ankr -32000 Unauthorized）按节点摘除
        语义处理——换节点重试，同一节点连续 3 次 auth error 即冷却摘除；
    (3) eth_getCode == "0x" 的文案一律为「EOA 或已自毁合约」，不得断言纯 EOA；
    (4) 所有启发式结论统一携带「启发式判断，非确定性结论，建议人工复核」文案；
        无法确认免费性的合约一律不给 ✅（评级引擎兜底为 ⚠️）。

实现说明:
    - 标准库无 keccak，所有函数 selector / 接口 ID 均为硬编码 hex 字面量
      （来源: ERC-165 规范与常见 NFT 合约 ABI 公开 selector；selector 若不匹配
      只会导致该次 eth_call 探测失败，无任何副作用，revert 不计红旗）。
    - 节点状态（连续失败计数、冷却截止时间）按 endpoint 维度隔离存放于
      NodePool（线程安全），不使用全局变量。
    - 任何单合约/单检查项的异常均被兜底捕获，不拖垮整体（AC-7）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# ============================================================================
# 常量区
# ============================================================================

RPC_ENDPOINTS: list[str] = [
    "https://bsc-dataseed.binance.org",
    "https://bsc-dataseed1.defibit.io",
    "https://rpc.ankr.com/bsc",
]

DEFAULT_WORKERS = 4          # 不变式(1)：默认 4 并发
DEFAULT_RPC_TIMEOUT = 10.0   # 不变式(1)：单次 RPC 超时 10s
DEFAULT_CONTRACT_BUDGET = 60.0  # 不变式(1)：单合约总预算 60s
MAX_RETRIES = 3              # 网络类错误重试次数（1+3=4 次请求上限）
BACKOFF_BASE = 0.5           # 退避序列 0.5 → 1 → 2（factor=2）
NODE_COOLDOWN_SECONDS = 60.0
NODE_FAILURE_THRESHOLD = 2   # 连续失败 ≥2 次 → 冷却摘除
AUTH_ERROR_THRESHOLD = 3     # 连续 auth 类 error ≥3 次 → 节点冷却摘除（第 2 轮修复）

# auth 类 JSON-RPC error 识别关键词（小写匹配 message，code 通常为 -32000）
AUTH_ERROR_KEYWORDS = ("unauthorized", "unauthorised", "authentication",
                       "auth failed", "api key", "api-key", "forbidden")

# 硬编码 selector（标准库无 keccak，来源见模块 docstring；
# 第 2 轮修复：全部 selector 经 keccak-256 本地重算并经 4byte.directory 交叉校验，
# 原表 mintPrice/maxSupply/maxMintAmount 三个 selector 为错值已修正）
SELECTORS: dict[str, str] = {
    # ERC-165（确定性）
    "supportsInterface": "0x01ffc9a7",
    # 常量探测（启发式；不匹配则探测失败，无副作用）
    "price()": "0xa035b1fe",
    "mintPrice()": "0x6817c76c",
    "cost()": "0x13faede6",
    "mintFee()": "0x13966db5",
    "fee()": "0xddca3f43",
    "maxSupply()": "0xd5abeb01",
    "maxMintAmount()": "0x239c70ae",
    "paused()": "0x5c975abb",
    "owner()": "0x8da5cb5b",
    "totalSupply()": "0x18160ddd",
}

# ERC-165 接口 ID（确定性）
INTERFACE_IDS: dict[str, str] = {
    "ERC-721": "0x80ac58cd",
    "ERC-1155": "0xd9b67a26",
}

# 字节码特征表: (特征名, 匹配子串(对 lower 后的 bytecode), 方向, 强度, 说明)
BYTECODE_PATTERNS: list[tuple[str, str, str, str, str]] = [
    ("selfdestruct_trace", "ff", "red_flag", "weak",
     "字节码含 0xff 特征，可能可自毁（弱信号，误报率高，需人工确认上下文）"),
    ("ownable_marker", "118c750a", "red_flag", "strong",
     "含 Ownable 权限模型痕迹（caller is not the owner 错误 selector），owner 可改规则"),
    ("blacklist_marker", "blacklist", "red_flag", "strong",
     "疑似含黑名单机制（blacklist/isBlacklisted ASCII 痕迹）"),
    ("pause_marker", "pause", "red_flag", "strong",
     "疑似可暂停 mint（paused/pause 函数名 ASCII 痕迹）"),
    ("mint_price_marker", "mintprice", "info", "strong",
     "含铸造价格变量痕迹（MINT_PRICE/mintPrice ASCII，正信号，辅助佐证 fee 探测）"),
    ("proxy_pattern",
     "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc",
     "red_flag", "strong",
     "EIP-1967 实现槽位痕迹，疑似代理合约，逻辑可被替换"),
    ("delegatecall_trace", "f4", "red_flag", "weak",
     "字节码含 delegatecall 操作码特征（弱信号，孤立出现误报高）"),
]

HEURISTIC_NOTICE = "启发式判断，非确定性结论，建议人工复核"
HEURISTIC_DISCLAIMER = ("特征匹配基于字节码模式，存在误报/漏报可能"
                        "（如代理合约、内联优化、非常量 price），结论仅供初筛。")

ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Entry:
    """有效输入条目。"""
    name: str
    address: str
    line_no: int


@dataclass
class InvalidEntry:
    """无效输入条目（原样记录，用于报告标注）。"""
    line_no: int
    raw: str
    reason: str


@dataclass
class CheckResult:
    """单个检查项结果。status: ok / warn / error / not_done"""
    check: str
    status: str
    conclusion: str
    evidence: str = ""
    heuristic: bool = False
    red_flag: bool = False        # 强红旗（strong 启发式或确定性红旗）
    weak_hits: list[str] = field(default_factory=list)


@dataclass
class ContractResult:
    entry: Entry
    results: dict[str, CheckResult] = field(default_factory=dict)
    level: str = "warn"           # risk / warn / ok / not_done
    reasons: list[str] = field(default_factory=list)


# ============================================================================
# RPC 模块：NodePool + failover
# ============================================================================

class NodePool:
    """3 节点池：按 endpoint 隔离健康状态，线程安全。

    摘除规则（04-算法设计 1.3 + 第 2 轮修复）:
      - 连续失败 ≥2 → cooldown_until = now + 60s，pick_node 跳过；
      - 节点级鉴权失败（auth 类 JSON-RPC error）连续 ≥3 次 → 同样冷却摘除
        （不变式 2 补充：业务层错误不换节点，但「节点级鉴权失败」按节点摘除处理）；
      - 冷却结束自动恢复轮换（自愈）；
      - 收到正常响应（含非 auth 类 JSON-RPC error）即重置失败计数；
      - 全部不健康时返回失败次数最少的节点（垂死挣扎）。
    """

    def __init__(self, endpoints: list[str]) -> None:
        self._lock = threading.Lock()
        self._endpoints = list(endpoints)
        # endpoint -> {"failures": int, "consecutive": int,
        #             "auth_consecutive": int, "cooldown_until": float}
        self._state: dict[str, dict[str, float | int]] = {
            ep: {"failures": 0, "consecutive": 0,
                 "auth_consecutive": 0, "cooldown_until": 0.0}
            for ep in endpoints
        }

    def pick_node(self) -> str | None:
        with self._lock:
            now = time.monotonic()
            healthy = [ep for ep in self._endpoints
                       if self._state[ep]["cooldown_until"] <= now]
            if healthy:
                return healthy[0]
            # 全部冷却：垂死挣扎，返回总失败次数最少的节点
            if self._endpoints:
                return min(self._endpoints,
                           key=lambda ep: self._state[ep]["failures"])
            return None

    def mark_success(self, endpoint: str) -> None:
        with self._lock:
            st = self._state[endpoint]
            st["consecutive"] = 0
            # 第 3 轮修复（QA）：auth_consecutive 不再被成功响应清零——
            # 鉴权失败是节点配置问题，间歇性成功不代表修复；
            # 累计达到阈值即摘除，进程生命周期内不重置。

    def mark_auth_failure(self, endpoint: str) -> bool:
        """节点级鉴权失败计数；连续 ≥AUTH_ERROR_THRESHOLD 次则冷却摘除。

        返回 True 表示本次触发了摘除（供上层日志/判断用）。
        """
        with self._lock:
            st = self._state[endpoint]
            st["auth_consecutive"] = int(st["auth_consecutive"]) + 1
            st["failures"] = int(st["failures"]) + 1
            if int(st["auth_consecutive"]) >= AUTH_ERROR_THRESHOLD:
                st["cooldown_until"] = max(
                    float(st["cooldown_until"]),
                    time.monotonic() + NODE_COOLDOWN_SECONDS)
                return True
            return False

    def mark_failure(self, endpoint: str, cooldown: float | None = None) -> None:
        with self._lock:
            st = self._state[endpoint]
            st["failures"] = int(st["failures"]) + 1
            st["consecutive"] = int(st["consecutive"]) + 1
            if int(st["consecutive"]) >= NODE_FAILURE_THRESHOLD or cooldown:
                until = time.monotonic() + (cooldown or NODE_COOLDOWN_SECONDS)
                st["cooldown_until"] = max(float(st["cooldown_until"]), until)


_POOL = NodePool(RPC_ENDPOINTS)   # 模块级单例池（状态按 endpoint 隔离，非裸全局变量语义）


def json_rpc_call(endpoint: str, method: str, params: list,
                  timeout: float = DEFAULT_RPC_TIMEOUT) -> dict:
    """对单个 endpoint 发起一次 JSON-RPC 调用。

    网络类错误（超时/URLError/HTTP 429/5xx）抛出异常由上层 failover 处理；
    节点正常应答（含 JSON-RPC error 体）原样返回 dict。
    """
    payload = json.dumps({"jsonrpc": "2.0", "id": 1,
                          "method": method, "params": params}).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=payload, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "bsc-mint-checker/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 429 or 500 <= e.code < 600:
            retry_after = e.headers.get("Retry-After") if e.headers else None
            err: Exception = TimeoutError(f"HTTP {e.code} rate-limit/5xx")
            setattr(err, "retry_after", retry_after)  # 供上层取 cooldown
            raise err from e
        raise
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("invalid JSON-RPC response")
    return data


def _is_auth_error(err: object) -> bool:
    """判定 JSON-RPC error 是否为节点级鉴权失败（如 Ankr -32000 Unauthorized）。"""
    if not isinstance(err, dict):
        return False
    msg = str(err.get("message", "")).lower()
    return any(kw in msg for kw in AUTH_ERROR_KEYWORDS)


def rpc_call_with_failover(method: str, params: list) -> tuple[str, object]:
    """带节点 failover + 指数退避的逻辑调用。

    返回 (kind, value):
      ("ok", result)      —— 成功
      ("rpc_error", err)  —— JSON-RPC error（业务层错误，不换节点不重试，不变式 2）
      ("error", msg)      —— 网络错误重试耗尽 / 节点池整体不可用
    退避计数按逻辑调用计（换节点也消耗重试次数），序列 0.5/1/2s ±20% 抖动。
    第 2 轮修复：auth 类 error（节点级鉴权失败）不计入业务错误，按节点摘除
    语义处理——计数换节点退避重试，同一节点连续 3 次即冷却摘除。
    """
    backoff = BACKOFF_BASE
    last_auth_err: object = None
    for attempt in range(MAX_RETRIES + 1):
        node = _POOL.pick_node()
        if node is None:
            return ("error", "节点池整体不可用")
        try:
            resp = json_rpc_call(node, method, params)
            _POOL.mark_success(node)
            if "error" in resp:
                err = resp["error"]
                if _is_auth_error(err):
                    _POOL.mark_auth_failure(node)
                    last_auth_err = err
                    if attempt < MAX_RETRIES:
                        time.sleep(backoff * random.uniform(0.8, 1.2))
                        backoff *= 2
                    continue  # 换节点重试（摘除由 NodePool 计数自动触发）
                return ("rpc_error", err)
            return ("ok", resp.get("result"))
        except Exception as e:  # 网络类错误：failover + 退避
            cooldown = None
            retry_after = getattr(e, "retry_after", None)
            if retry_after:
                try:
                    cooldown = max(0.0, float(retry_after))
                except (TypeError, ValueError):
                    cooldown = None
            _POOL.mark_failure(node, cooldown=cooldown)
            if attempt < MAX_RETRIES:
                time.sleep(backoff * random.uniform(0.8, 1.2))
                backoff *= 2
    if last_auth_err is not None:
        return ("rpc_error", last_auth_err)
    return ("error", f"重试 {MAX_RETRIES} 次后仍失败")


# ============================================================================
# 检查器
# ============================================================================

def decode_uint(hexdata: str) -> int | None:
    """把 eth_call 返回的 32 字节 return data 解码为 uint，失败返回 None。"""
    if not isinstance(hexdata, str) or not hexdata.startswith("0x"):
        return None
    h = hexdata[2:].lower()
    if h == "":
        return None
    if len(h) > 64:
        h = h[:64]  # 只取第一个 word
    if len(h) < 64:
        return None
    try:
        return int(h, 16)
    except ValueError:
        return None


def eth_call_const(to_addr: str, selector: str,
                   arg: str = "") -> tuple[str, object]:
    """eth_call 常量查询，arg 为已 pad 到 32 字节的 hex（无 0x 前缀）。"""
    data = selector + arg
    return rpc_call_with_failover(
        "eth_call", [{"to": to_addr, "data": data}, "latest"])


def check_existence(addr: str) -> CheckResult:
    """eth_getCode 判定（确定性）。不变式(3)：0x 文案为「EOA 或已自毁合约」。"""
    kind, val = rpc_call_with_failover("eth_getCode", [addr, "latest"])
    if kind != "ok":
        return CheckResult("存在性", "error", f"检查未完成: {val}")
    code = val if isinstance(val, str) else "0x"
    if code == "0x":
        return CheckResult(
            "存在性", "ok", "该地址无链上代码（EOA，或合约已自毁）",
            evidence="eth_getCode → 0x")
    size = (len(code) - 2) // 2
    return CheckResult(
        "存在性", "ok", "合约",
        evidence=f"eth_getCode → 0x…{len(code)} hex chars（≈{size} 字节）")


def check_interface(addr: str) -> CheckResult:
    """ERC-165 supportsInterface 识别类型（确定性）。"""
    found: list[str] = []
    evid: list[str] = []
    for name, iid in INTERFACE_IDS.items():
        kind, val = eth_call_const(addr, SELECTORS["supportsInterface"],
                                   iid[2:].lower().ljust(64, "0"))
        if kind == "ok" and isinstance(val, str):
            supported = decode_uint(val) == 1
            evid.append(f"{name}({iid}) → {val[:10]}{'…' if len(val) > 10 else ''} "
                        f"({'支持' if supported else '不支持'})")
            if supported:
                found.append(name)
    if not evid:
        return CheckResult(
            "接口类型", "warn",
            "无法识别合约类型（ERC-165 探测失败或合约未实现 supportsInterface）",
            evidence="; ".join(evid) or "全部 eth_call 探测失败", heuristic=True)
    ctype = "/".join(found) if found else "非标准 ERC-721/1155"
    return CheckResult("接口类型", "ok", ctype, evidence="; ".join(evid))


def _probe_consts(addr: str, names: list[str]) -> tuple[dict[str, int], list[str]]:
    """对给定 selector 名单做 eth_call 探测，返回 (成功解码值, 证据行)。"""
    values: dict[str, int] = {}
    evid: list[str] = []
    for name in names:
        kind, val = eth_call_const(addr, SELECTORS[name])
        if kind == "ok" and isinstance(val, str) and val not in ("0x", "0x00"):
            num = decode_uint(val)
            if num is not None:
                values[name] = num
                evid.append(f"{name} = {num}（wei）")
                continue
        evid.append(f"{name} 探测失败（未实现或 revert，非红旗）")
    return values, evid


def check_fee_heuristics(addr: str) -> CheckResult:
    """铸造价格探测（启发式）：任一价格 getter 探测到 0 才算「探测确认免费」。

    第 2 轮修复：补充免费档常见 getter（cost/mintFee/fee），并修正
    mintPrice/maxSupply/maxMintAmount 的错误 selector（QA 5.x 探测失效根因）。
    """
    values, evid = _probe_consts(
        addr, ["price()", "mintPrice()", "cost()", "mintFee()", "fee()"])
    zeros = sorted(n for n, v in values.items() if v == 0)
    nonzeros = {n: v for n, v in values.items() if v > 0}
    if zeros:
        return CheckResult(
            "铸造费用", "ok",
            f"探测确认铸造价格为 0（免费）（getter: {', '.join(zeros)}）",
            evidence="; ".join(evid), heuristic=True)
    if nonzeros:
        detail = ", ".join(f"{n}={v} wei" for n, v in sorted(nonzeros.items()))
        return CheckResult(
            "铸造费用", "warn", f"探测到非零铸造价格: {detail}",
            evidence="; ".join(evid), heuristic=True, red_flag=True)
    return CheckResult(
        "铸造费用", "warn", "无法确认免费性（price 探测失败/非常量/代理）",
        evidence="; ".join(evid) or "无", heuristic=True)


def check_limit_heuristics(addr: str) -> CheckResult:
    """铸造限制探测（启发式）。"""
    values, evid = _probe_consts(addr, ["maxSupply()", "maxMintAmount()"])
    if not values:
        return CheckResult(
            "铸造限制", "warn", "未探测到铸造限制常量",
            evidence="; ".join(evid) or "无", heuristic=True)
    return CheckResult(
        "铸造限制", "ok", "探测到铸造限制常量（启发式参考）",
        evidence="; ".join(evid), heuristic=True)


def check_bytecode_flags(code_hex: str) -> CheckResult:
    """字节码特征扫描（启发式，strong/weak 两级）。

    第 2 轮修复（QA 4.2）：0xff/0xf4 等 1 字节弱信号实测 22/22 全命中、零区分度，
    故降级为「仅在无其他信号时提示」——weak 命中永不单独/叠加触发 warn 红旗，
    仅在无 strong 命中时作为中性信息展示；strong 命中 → warn + red_flag
    （04-算法设计 4.3 的「多条 weak 叠加成 warn」规则随之废止）。
    """
    code = code_hex.lower() if isinstance(code_hex, str) else ""
    strong_hits: list[str] = []
    weak_hits: list[str] = []
    info_hits: list[str] = []
    for name, pattern, direction, strength, desc in BYTECODE_PATTERNS:
        if pattern in code:
            snippet = code[max(0, code.find(pattern)):][:32]
            line = f"{name}: {desc}（命中片段 {snippet}…）"
            if direction == "red_flag" and strength == "strong":
                strong_hits.append(line)
            elif direction == "red_flag" and strength == "weak":
                weak_hits.append(line)
            elif direction == "info":
                info_hits.append(line)
            else:  # red_flag + medium（本表当前未用，兜底归 strong）
                strong_hits.append(line)
    if not strong_hits:
        # 无 strong 命中：weak 仅作中性提示（不触发 warn/red_flag）
        notes = weak_hits + info_hits
        if not notes:
            return CheckResult(
                "字节码特征", "ok", "未命中任何已知风险特征",
                evidence=f"bytecode 长度 {len(code)} hex chars（启发式扫描）",
                heuristic=True)
        return CheckResult(
            "字节码特征", "ok",
            "仅弱信号提示（不计入风险，孤立出现误报率高）: " + "; ".join(notes)
            + "；" + HEURISTIC_DISCLAIMER,
            evidence="; ".join(notes), heuristic=True, weak_hits=weak_hits)
    merged = "; ".join(strong_hits)
    if weak_hits:
        merged += f"（另含 {len(weak_hits)} 条弱信号，不计入风险）"
    return CheckResult(
        "字节码特征", "warn", merged + "；" + HEURISTIC_DISCLAIMER,
        evidence="; ".join(strong_hits + weak_hits + info_hits),
        heuristic=True, red_flag=True, weak_hits=weak_hits)


def check_bscscan_verified(addr: str) -> CheckResult:
    """可选项：配置 BSCSCAN_API_KEY 时做 getsourcecode 开源验证。

    第 2 轮修复（QA AC-4）：无 key 时不再写「跳过」，改为明确的
    「未验证（无 API key），风险未知，按 ⚠️ 计入建议」结论（status=warn）。
    """
    key = os.environ.get("BSCSCAN_API_KEY", "")
    if not key:
        return CheckResult(
            "BscScan 开源验证", "warn",
            "未验证（无 BSCSCAN_API_KEY），开源状态未知，风险未知，按 ⚠️ 计入建议",
            evidence="环境变量 BSCSCAN_API_KEY 未配置")
    url = ("https://api.bscscan.com/api?module=contract&action=getsourcecode"
           f"&address={addr}&apikey={key}")
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=DEFAULT_RPC_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        items = data.get("result") if isinstance(data, dict) else None
        if isinstance(items, list) and items:
            src = (items[0].get("SourceCode") or "").strip()
            abi = (items[0].get("ABI") or "").strip()
            verified = bool(src) and abi not in ("", "-")
            return CheckResult(
                "BscScan 开源验证", "ok",
                "已开源验证" if verified else "未开源（或源码为空）",
                evidence=f"ABI 字段: {abi[:20]}")
        return CheckResult("BscScan 开源验证", "error",
                           f"BscScan 返回异常: {str(data)[:80]}")
    except Exception as e:  # 可选项失败不阻塞主流程
        return CheckResult("BscScan 开源验证", "error",
                           f"BscScan 不可达: {e}")


# ============================================================================
# 输入解析 / 校验
# ============================================================================

def validate_address(addr: str) -> bool:
    """0x + 40 位 hex（EIP-55 校验和校验需 keccak，标准库不做，仅格式校验）。"""
    return bool(ADDR_RE.match(addr.strip()))


def parse_contracts_file(path: str) -> tuple[list[Entry], list[InvalidEntry]]:
    """解析「名称,地址」清单；空行忽略；非法行原样记录。"""
    entries: list[Entry] = []
    invalid: list[InvalidEntry] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue  # 空行忽略（AC-1）
            if "," not in line:
                invalid.append(InvalidEntry(i, line, "缺少逗号分隔的「名称,地址」格式"))
                continue
            name, _, addr = line.partition(",")
            name, addr = name.strip(), addr.strip()
            if not validate_address(addr):
                invalid.append(InvalidEntry(i, line, f"地址格式非法: {addr[:46]}"))
                continue
            entries.append(Entry(name=name or "(未命名)", address=addr, line_no=i))
    return entries, invalid


# ============================================================================
# 评级引擎 + 管线
# ============================================================================

def rate_contract(results: dict[str, CheckResult]) -> tuple[str, list[str]]:
    """规则引擎（04-算法设计 5.5）。

    优先级：确定性红旗 → ❌(risk)；启发式疑点 → ⚠️(warn)；
    无红旗且探测确认免费 → ✅(ok)；其余一律 ⚠️ 兜底。
    核心不变式(4)：无法确认免费性 ⇒ 禁止 ✅。
    """
    reasons: list[str] = []
    free_confirmed = False
    has_blocker = False       # 确定性红旗
    has_warn = False          # 启发式疑点
    has_error = False

    # 第 3 轮修复（QA P0）：results 以 fn.__name__ 为键（如 "check_existence"），
    # 原实现用中文键取值永远取到 None，导致评级恒为 ⚠️ 兜底。
    exist = results.get("check_existence")
    if exist is not None and exist.status == "ok" and "无链上代码" in exist.conclusion:
        has_blocker = True
        reasons.append("目标地址不是链上合约（EOA 或已自毁合约），无法进行 mint 体检")
    elif exist is not None and exist.status == "error":
        has_error = True
        reasons.append(f"[存在性] {exist.conclusion}")

    fee = results.get("check_fee_heuristics")
    if fee is not None:
        if "探测确认铸造价格为 0" in fee.conclusion:
            free_confirmed = True
        elif fee.status == "warn":
            has_warn = True
            reasons.append(fee.conclusion)
        elif fee.status == "error":
            has_error = True

    # 第 3 轮修复（QA）：red_flag 由评级引擎显式消费（确定性/强启发式红旗 → warn）
    for key, label in (
        ("check_bytecode_flags", "字节码特征"),
        ("check_interface", "接口类型"),
        ("check_limit_heuristics", "铸造限制"),
        ("check_bscscan_verified", "BscScan 开源验证"),
    ):
        r = results.get(key)
        if r is None:
            continue
        if r.status == "warn":
            has_warn = True
            reasons.append(f"[{label}] {r.conclusion}")
        elif r.status == "error":
            has_error = True
        if getattr(r, "red_flag", False) and key != "check_fee_heuristics":
            has_warn = True

    if has_blocker:
        return ("risk", reasons)
    if not free_confirmed:
        # 不变式(4)：无法确认免费 ⇒ 不给 ✅
        if has_error:
            reasons.append("存在检查项失败，无法确认免费性")
        else:
            reasons.append("未能探测确认铸造价格为 0，无法确认免费性")
        return ("warn", reasons)
    if has_warn:
        return ("warn", reasons)
    return ("ok", ["探测确认铸造价格为 0，且未命中确定性/强启发式红旗"])


CHECK_PIPELINE: list = [
    check_existence,
    check_interface,
    check_fee_heuristics,
    check_limit_heuristics,
    check_bytecode_flags,
    check_bscscan_verified,
]


def check_one_contract(entry: Entry, budget: float,
                       timeout: float) -> ContractResult:
    """单合约检查管线（并发 worker 单元，异常全兜底，AC-7）。"""
    t0 = time.monotonic()
    cr = ContractResult(entry=entry)
    for fn in CHECK_PIPELINE:
        if time.monotonic() - t0 > budget:  # 单合约总预算
            cr.results[fn.__name__] = CheckResult(
                fn.__name__.replace("check_", ""), "not_done", "超时预算，未执行")
            continue
        try:
            if fn is check_bytecode_flags:
                # 需要先取 code；管线内单独调用 getCode
                kind, val = rpc_call_with_failover(
                    "eth_getCode", [entry.address, "latest"])
                if kind == "ok" and isinstance(val, str) and val != "0x":
                    cr.results[fn.__name__] = fn(val)
                elif kind == "ok":
                    cr.results[fn.__name__] = CheckResult(
                        "字节码特征", "ok", "无字节码（EOA/已自毁），跳过特征扫描",
                        evidence="eth_getCode → 0x")
                else:
                    cr.results[fn.__name__] = CheckResult(
                        "字节码特征", "error", f"获取字节码失败: {val}")
            else:
                cr.results[fn.__name__] = fn(entry.address)
        except Exception as e:  # 单点失败不中断（AC-7）
            cr.results[fn.__name__] = CheckResult(
                fn.__name__.replace("check_", ""), "error", f"异常: {e}")
    cr.level, cr.reasons = rate_contract(cr.results)
    return cr


def main_pipeline(entries: list[Entry], workers: int, budget: float,
                  timeout: float) -> list[ContractResult]:
    """合约级 4 并发（不变式 1），节点级 failover 串行轮换，不做请求级并发。"""
    if not entries:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda e: check_one_contract(e, budget, timeout),
                             entries))


# ============================================================================
# 报告生成
# ============================================================================

LEVEL_META: dict[str, tuple[str, str]] = {
    "risk": ("❌", "高风险"),
    "warn": ("⚠️", "存疑"),
    "ok": ("✅", "通过"),
    "not_done": ("⏳", "未完成"),
}
# 第 3 轮修复（QA 复验）：恢复 PRD AC-5 原定义「风险从低到高」（✅ → ⚠️ → ⏳ → ❌），
# 便于用户一眼看到可先铸的合约（用户故事 3）。第 2 轮的反向改动作废；
# 同级内 sorted() 为稳定排序，results 由 pool.map 生成天然保持输入顺序。
LEVEL_ORDER = {"ok": 0, "warn": 1, "not_done": 2, "risk": 3}


def _type_of(cr: ContractResult) -> str:
    r = cr.results.get("check_interface")
    return r.conclusion if r else "未知"


def _free_of(cr: ContractResult) -> str:
    fee = cr.results.get("check_fee_heuristics")
    if fee and "探测确认铸造价格为 0" in fee.conclusion:
        return "✅ 免费（探测确认）"
    if fee and "非零铸造价格" in fee.conclusion:
        return "❌ 非免费（探测到价格）"
    return "⚠️ 未能确认"


def _flags_of(cr: ContractResult) -> str:
    if cr.level == "risk":
        return cr.reasons[0][:60] if cr.reasons else "确定性红旗"
    if cr.level == "warn":
        return cr.reasons[0][:60] if cr.reasons else "启发式疑点"
    if cr.level == "not_done":
        return "检查未完成"
    return "无"


def generate_report(entries: list[Entry], invalid: list[InvalidEntry],
                    results: list[ContractResult],
                    out_path: str | None) -> str:
    """生成 markdown 报告（返回报告全文；out_path 为 None 时仅返回文本）。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    stats: dict[str, int] = {"risk": 0, "warn": 0, "ok": 0, "not_done": 0}
    for cr in results:
        stats[cr.level] = stats.get(cr.level, 0) + 1
    lines: list[str] = []
    ap = lines.append
    ap("# BSC 免费 mint 合约批量体检报告")
    ap("")
    ap(f"- 生成时间: {now}")
    ap(f"- RPC 节点: {', '.join(RPC_ENDPOINTS)}")
    ap(f"- 检查总数: {len(entries)}（无效输入 {len(invalid)} 条）")
    ap(f"- 结果分布: ❌ 高风险 {stats['risk']} | ⚠️ 存疑 {stats['warn']} | "
       f"✅ 通过 {stats['ok']} | ⏳ 未完成 {stats['not_done']}")
    ap("")
    if invalid:
        ap("## 无效输入")
        ap("")
        ap("| 行号 | 原始内容 | 原因 |")
        ap("|---|---|---|")
        for iv in invalid:
            ap(f"| {iv.line_no} | `{iv.raw[:60]}` | {iv.reason} |")
        ap("")
    ap("## 汇总表（按风险从低到高排序，✅ 可先铸在前）")
    ap("")
    ap("| 名称 | 地址 | 类型 | 风险 | 建议 |")
    ap("|---|---|---|---|---|")
    ordered = sorted(results, key=lambda c: LEVEL_ORDER.get(c.level, 9))
    for cr in ordered:
        icon, label = LEVEL_META[cr.level]
        ap(f"| {cr.entry.name} | `{cr.entry.address}` | {_type_of(cr)} | "
           f"{icon} {label} | {_flags_of(cr)} |")
    ap("")
    ap("## 逐合约详情")
    ap("")
    for cr in ordered:
        icon, label = LEVEL_META[cr.level]
        ap(f"### {icon} {cr.entry.name} (`{cr.entry.address}`)")
        ap("")
        ap(f"**综合评级**: {icon} {label}")
        if cr.reasons:
            ap("")
            for r in cr.reasons:
                ap(f"- {r}")
        ap("")
        for name, r in cr.results.items():
            status_icon = {"ok": "✅", "warn": "⚠️",
                           "error": "❌", "not_done": "⏳"}.get(r.status, "?")
            ap(f"- **{name}**: {status_icon} {r.conclusion}")
            if r.evidence:
                ap(f"  - 证据: `{r.evidence}`")
            if r.heuristic:
                ap(f"  - ⚠️ {HEURISTIC_NOTICE}")
        ap("")
    ap("## 免责声明")
    ap("")
    ap("本报告为自动化初筛结果，非安全审计，不构成投资建议；"
       "所有启发式结论需人工复核。" + HEURISTIC_DISCLAIMER)
    ap("")
    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def print_summary(report_path: str, results: list[ContractResult]) -> None:
    """终端一句话总结。"""
    stats: dict[str, int] = {}
    for cr in results:
        stats[cr.level] = stats.get(cr.level, 0) + 1
    dest = report_path if report_path else "stdout"
    print(f"[bsc-mint-checker] 共 {len(results)} 个合约: "
          f"❌{stats.get('risk', 0)} ⚠️{stats.get('warn', 0)} "
          f"✅{stats.get('ok', 0)} ⏳{stats.get('not_done', 0)}；"
          f"报告已输出到 {dest}")


# ============================================================================
# 入口
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_mint.py",
        description="BSC 免费 mint 合约批量体检 CLI（仅标准库）")
    parser.add_argument("input", help="输入清单文件（每行「名称,地址」）")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 markdown 报告路径（默认 stdout）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        choices=range(1, 9), metavar="1-8",
                        help=f"并发数（默认 {DEFAULT_WORKERS}，范围 1~8）")
    args = parser.parse_args(argv)

    try:
        entries, invalid = parse_contracts_file(args.input)
    except OSError as e:
        print(f"错误: 无法读取输入文件 {args.input}: {e}", file=sys.stderr)
        return 2

    try:
        results = main_pipeline(entries, workers=args.workers,
                                budget=DEFAULT_CONTRACT_BUDGET,
                                timeout=DEFAULT_RPC_TIMEOUT)
        report = generate_report(entries, invalid, results, args.output)
        if args.output is None:
            print(report)
        print_summary(args.output or "", results)
        return 0
    except Exception as e:  # 顶层兜底：任何意外不崩溃，产出错误信息
        print(f"错误: 执行中断: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
