#!/usr/bin/env python3
"""GPU 快照采集与管控（交付物 A）。

JSON 顶层键（schema v1，冻结）：schema_version, mode, host, collected_at, nvidia_smi,
driver_version, thresholds{vram_warn,temp_warn,util_warn}, gpus[], alerts[], warnings[],
placement。gpus[] 每项：index,name,uuid,vram_total_mb,vram_used_mb,vram_free_mb,
util_percent,temp_c,power_draw_w,power_limit_w,processes[],alerts[]。
退出码（优先级 3>2>1>0）：0 正常 / 1 有告警 / 2 采集失败 / 3 参数错误。
分层：argparse -> 纯函数内核（parse_* / evaluate_alerts / build_plan / render_*）-> main。
纯函数内禁止 print / open / subprocess / 读时间戳；subprocess 仅在采集与输出段。
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
EXIT_OK, EXIT_ALERTS, EXIT_COLLECT, EXIT_USAGE = 0, 1, 2, 3   # 全文件唯一出处

# 阈值单一出处：argparse 默认值、判定、渲染、JSON 全部引用此表
DEFAULTS = {"vram_warn": 90.0, "temp_warn": 85.0, "util_warn": 90.0}
WATERMARK = 0.9          # 分配安全水位，预留 10% 给峰值与碎片
SENTINEL_OFF = 0.0       # 任一阈值传 0 = 关闭该项
MIB_PER_GIB = 1024       # GiB -> MiB 唯一换算常量
TIMEOUT_S = 15
TS_FORMAT = "%Y%m%d-%H%M%S"

FIELDS_CORE = ["index", "uuid", "name", "driver_version", "memory.total", "memory.used",
               "memory.free", "utilization.gpu", "temperature.gpu", "power.draw",
               "power.limit"]
FIELDS_MIN = ["index", "name", "memory.total", "memory.used"]
FIELDS_APPS = ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
# 远端命令：模块级字面量，全文件仅出现 2 次（定义 1 + 引用 1），零插值
SMI_REMOTE_CMD = (
    "nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,"
    "memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit "
    "--format=csv,noheader,nounits")
# mock 是「一段 CSV 文本」，与线上逐字节同构，走完全相同的解析器（不变式 2）
MOCK_SMI_TEXT = """\
0, GPU-4f2a1c9e-mock-0001, NVIDIA GeForce RTX 4090, 550.54.14, 24564, 21600, 2964, 88, 74, 320.42, 450.00
1, GPU-7b3d5e8a-mock-0002, NVIDIA GeForce RTX 3090, 550.54.14, 24576, 512, 24064, 0, 34, 68.15, 350.00
"""
MOCK_APPS_TEXT = """\
GPU-4f2a1c9e-mock-0001, 24817, /usr/bin/python3, 11200
GPU-4f2a1c9e-mock-0001, 24903, /usr/bin/python3, 10300
"""

HINTS_NO_SMI = (
    "安装 NVIDIA 驱动（nvidia-smi 随驱动提供），装完执行 nvidia-smi -L 自检",
    "设置环境变量 NVIDIA_SMI_PATH=/usr/bin/nvidia-smi，或用 --nvidia-smi <path> 显式指定",
    "本机无卡时改用 --host <主机> 采集远端，或用 --mock 演示")
HINTS_REMOTE = (
    "确认主机名可解析、已配置 ssh 免密（BatchMode=yes），先手工 ssh <host> nvidia-smi 验证",
    "多主机/跳板机请在 ~/.ssh/config 用 ProxyJump 配置别名，本工具只接单主机",
    "远端无卡或驱动异常时改用 --mock 演示链路")

_FIELD_ERR_RE = re.compile(r'Field "(\w+\.\w+)" is not a valid field')
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_NUM_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*(mib|mb|gib|gb|%|c|w|mhz)?$", re.I)
_SENTINELS = {"n/a", "[n/a]", "not supported", "[not supported]", "unknown", "[unknown error]"}
_DECIMAL_UNIT = re.compile(r"\b(mb|gb)\b", re.I)
_HOST_RE = re.compile(r"[A-Za-z0-9._-]{1,253}")

class UsageError(Exception):
    """参数错误 -> exit 3。"""

    def __init__(self, message: str, hints: Tuple[str, ...] = ()):
        super().__init__(message)
        self.message, self.hints = message, hints

class CollectError(Exception):
    """采集失败 -> exit 2。"""

    def __init__(self, message: str, hints: Tuple[str, ...] = ()):
        super().__init__(message)
        self.message, self.hints = message, hints

@dataclass
class Proc:
    pid: Optional[int]
    name: str
    used_mb: Optional[float]

@dataclass
class Alert:
    gpu_index: int
    kind: str          # "vram" | "temp" | "util"（闭集，渲染层按此分发）
    value: float
    threshold: float
    message: str

@dataclass
class Gpu:
    index: int
    name: str
    uuid: Optional[str] = None
    vram_total_mb: Optional[float] = None; vram_used_mb: Optional[float] = None
    vram_free_mb: Optional[float] = None; util_percent: Optional[float] = None
    temp_c: Optional[float] = None
    power_draw_w: Optional[float] = None; power_limit_w: Optional[float] = None
    processes: List[Proc] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)     # 与顶层 alerts 双写

@dataclass
class Thresholds:
    vram_warn: float; temp_warn: float; util_warn: float

@dataclass
class Snapshot:
    schema_version: int
    mode: str            # "local" | "remote" | "mock" | "csv"
    host: str
    collected_at: str
    nvidia_smi: Optional[str]
    driver_version: Optional[str]
    gpus: List[Gpu]
    alerts: List[Alert]
    warnings: List[str]
    thresholds: Thresholds
    placement: Optional[Dict[str, object]] = None

def _warn(bucket: Optional[List[str]], msg: str) -> None:
    if bucket is not None:
        bucket.append(msg)

def _num(v: Optional[float], d: int = 2) -> str:
    return "n/a" if v is None else ("%." + str(d) + "f") % v

def _threshold(v: float, unit: str) -> str:
    return "关闭" if v <= SENTINEL_OFF else ("%g" % v) + unit

def parse_float(raw: Optional[str]) -> Optional[float]:
    """字段归一：哨兵词/空 -> None；剥单位后转 float。任何失败返回 None，绝不返回 0。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in _SENTINELS:
        return None
    m = _NUM_RE.match(text)
    if not m:
        return None
    return float(m.group(1)) * (MIB_PER_GIB if (m.group(2) or "").lower() in ("gib", "gb") else 1)

def split_row(line: str, fields: List[str]) -> Optional[List[str]]:
    """按字段名定位切分；多余段并回 name（GPU 名含逗号），少段返回 None（该行跳过）。"""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < len(fields):
        return None
    if len(parts) > len(fields):
        pos = fields.index("name")
        tail = len(fields) - pos - 1
        parts = (parts[:pos] + [",".join(parts[pos:len(parts) - tail]).strip()]
                 + (parts[len(parts) - tail:] if tail else []))
    return parts

def parse_smi_csv(text: str, fields: List[str],
                  warnings: Optional[List[str]] = None) -> List[Gpu]:
    """主解析入口：nvidia-smi 的 csv stdout -> List[Gpu]；单行坏数据不拖垮整表。"""
    gpus: List[Gpu] = []
    for raw in text.replace("\r", "\n").splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("index"):
            continue
        parts = split_row(line, fields)
        if parts is None:
            _warn(warnings, "字段数少于期望，已跳过该行：" + line[:60])
            continue
        row = dict(zip(fields, parts))
        idx = parse_float(row.get("index"))
        if idx is None:
            _warn(warnings, "index 不可解析，已跳过该行：" + line[:60])
            continue
        label = "GPU%d" % int(idx)
        total, used = parse_float(row.get("memory.total")), parse_float(row.get("memory.used"))
        free = parse_float(row.get("memory.free"))
        util, temp = parse_float(row.get("utilization.gpu")), parse_float(row.get("temperature.gpu"))
        if util is not None and not 0.0 <= util <= 100.0:       # 驱动偶发报 101
            util = min(max(util, 0.0), 100.0)
            _warn(warnings, label + " 利用率越界，已截断到 [0,100]")
        if temp is not None and not -40.0 <= temp <= 150.0:     # 0 常见于传感器不可读
            temp = None
            _warn(warnings, label + " 温度越界，已置 null（传感器不可读）")
        if total and used is not None and used > total:
            _warn(warnings, "%s 已用显存大于总量（%s > %s MiB），按原值保留" % (label, used, total))
        if total is not None and used is not None:
            derived = total - used
            if free is None:
                free = derived
                _warn(warnings, label + " 缺少 memory.free，已按 total-used 补算")
            elif abs(free - derived) > 0.05 * total:
                free = derived
                _warn(warnings, label + " memory.free 与 total-used 偏差超 5%，以后者为准")
        gpus.append(Gpu(int(idx), (row.get("name") or "").strip() or "unknown",
                        (row.get("uuid") or "").strip() or None, total, used, free, util,
                        temp, parse_float(row.get("power.draw")),
                        parse_float(row.get("power.limit"))))
    if warnings is not None and _DECIMAL_UNIT.search(text):
        _warn(warnings, "字段使用十进制单位 MB/GB，已按 MiB 解释（偏高不超过 4.8%）")
    return gpus

def parse_driver_version(text: str, fields: List[str]) -> Optional[str]:
    """取首行的 driver_version；不可用 -> None。"""
    if "driver_version" not in fields:
        return None
    for raw in text.replace("\r", "\n").splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("index"):
            continue
        parts = split_row(line, fields)
        if parts is not None:
            value = (dict(zip(fields, parts)).get("driver_version") or "").strip()
            return value if value and value.lower() not in _SENTINELS else None
    return None

def parse_compute_apps(text: str) -> Dict[str, List[Proc]]:
    """按 gpu_uuid 归集进程；No running processes found -> 空 dict（不抛异常）。"""
    out: Dict[str, List[Proc]] = {}
    body = text.replace("\r", "\n")
    if "No running processes found" in body:
        return out
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("gpu_uuid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(FIELDS_APPS):
            continue
        pid = parse_float(parts[1])
        out.setdefault(parts[0], []).append(
            Proc(int(pid) if pid is not None else None,
                 ",".join(parts[2:len(parts) - 1]).strip() or "unknown",
                 parse_float(parts[-1])))
    return out

def attach_processes(gpus: List[Gpu], apps: Dict[str, List[Proc]]) -> None:
    """uuid 命中即挂；无 uuid 时按出现顺序退回 index。"""
    keys = list(apps.keys())
    for pos, gpu in enumerate(gpus if keys else []):
        procs = apps.get(gpu.uuid) if gpu.uuid else None
        if procs is None and len(keys) == len(gpus) and pos < len(keys):
            procs = apps.get(keys[pos])
        if procs:
            gpu.processes = list(procs)

def evaluate_alerts(gpus: List[Gpu], th: Thresholds) -> List[Alert]:
    """三类告警各自独立成条，>= 触发，阈值 0 = 关闭，按 (index, kind) 升序。"""
    out: List[Alert] = []
    for g in gpus:
        label = "GPU%d" % g.index
        if th.vram_warn > SENTINEL_OFF and g.vram_total_mb and g.vram_used_mb is not None:
            pct = min(g.vram_used_mb / g.vram_total_mb * 100.0, 100.0)
            if pct >= th.vram_warn:
                out.append(Alert(g.index, "vram", round(pct, 2), th.vram_warn,
                                 "%s 显存 %.2f%% >= %g%%，仅剩 %d MiB"
                                 % (label, pct, th.vram_warn, int(g.vram_free_mb or 0.0))))
        if th.temp_warn > SENTINEL_OFF and g.temp_c is not None and g.temp_c >= th.temp_warn:
            out.append(Alert(g.index, "temp", g.temp_c, th.temp_warn,
                             "%s 温度 %g 摄氏度 >= %g 摄氏度" % (label, g.temp_c, th.temp_warn)))
        if (th.util_warn > SENTINEL_OFF and g.util_percent is not None
                and g.util_percent >= th.util_warn):
            out.append(Alert(g.index, "util", g.util_percent, th.util_warn,
                             "%s 利用率 %g%% >= %g%%，卡上有 %d 个进程"
                             % (label, g.util_percent, th.util_warn, len(g.processes))))
    out.sort(key=lambda a: (a.gpu_index, a.kind))
    return out

def build_plan(gpus: List[Gpu], need_gb: float, watermark: float,
               th: Thresholds) -> Dict[str, object]:
    """分配建议：avail = floor(total*watermark) - used；fit 需不 busy / 不 hot。"""
    need_mb = need_gb * MIB_PER_GIB
    cands: List[Dict[str, object]] = []
    for g in gpus:
        busy = (th.util_warn > SENTINEL_OFF and g.util_percent is not None
                and g.util_percent >= th.util_warn)
        hot = th.temp_warn > SENTINEL_OFF and g.temp_c is not None and g.temp_c >= th.temp_warn
        if g.vram_total_mb is None or g.vram_used_mb is None:
            cands.append({"index": g.index, "name": g.name, "avail_mb": None,
                          "after_used_pct": None, "fit": False, "reason": "no_vram_info"})
            continue
        avail = math.floor(g.vram_total_mb * watermark) - g.vram_used_mb
        reason = ("insufficient" if need_mb > avail
                  else ("busy" if busy else ("hot" if hot else "fit")))
        after = (g.vram_used_mb + need_mb) / g.vram_total_mb * 100.0
        cands.append({"index": g.index, "name": g.name, "avail_mb": int(avail),
                      "after_used_pct": round(after, 2), "fit": reason == "fit",
                      "reason": reason})
    fitted = sorted([c for c in cands if c["fit"]],
                    key=lambda c: (c["after_used_pct"], c["index"]))
    for rank, cand in enumerate(fitted):
        cand["rank"] = rank + 1
    rest = sorted([c for c in cands if not c["fit"]], key=lambda c: c["index"])
    return {"need_gb": need_gb, "need_mb": int(need_mb), "watermark": watermark,
            "recommended": fitted[0]["index"] if fitted else None,
            "candidates": fitted + rest}

def _alert_dict(a: Alert) -> Dict[str, object]:
    return {"gpu_index": a.gpu_index, "kind": a.kind, "value": a.value,
            "threshold": a.threshold, "message": a.message}

def render_markdown(snap: Snapshot) -> str:
    th = snap.thresholds
    out = ["# GPU 快照报告", "",
           "- 模式: `%s`" % snap.mode, "- 主机: `%s`" % snap.host,
           "- 采集时间: `%s`" % snap.collected_at,
           "- nvidia-smi: `%s`" % (snap.nvidia_smi or "n/a"),
           "- 驱动版本: `%s`" % (snap.driver_version or "n/a"),
           "- 告警阈值: 显存 %s / 温度 %s / 利用率 %s（0 = 关闭）"
           % (_threshold(th.vram_warn, "%"), _threshold(th.temp_warn, "℃"),
              _threshold(th.util_warn, "%")),
           "- schema_version: %d" % snap.schema_version, "",
           "## GPU 概览", "",
           "| idx | 名称 | 总显存 MiB | 已用 MiB | 空闲 MiB | 占用率 | 利用率 | 温度 "
           "| 功耗 W | 进程 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in snap.gpus:
        pct = ("%.2f%%" % min(g.vram_used_mb / g.vram_total_mb * 100.0, 100.0)
               if g.vram_total_mb and g.vram_used_mb is not None else "n/a")
        out.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s / %s | %d |" % (
            g.index, g.name, _num(g.vram_total_mb, 0), _num(g.vram_used_mb, 0),
            _num(g.vram_free_mb, 0), pct, _num(g.util_percent, 0), _num(g.temp_c, 0),
            _num(g.power_draw_w), _num(g.power_limit_w), len(g.processes)))
    procs = ["| %d | %s | %s | %s |" % (g.index, "n/a" if p.pid is None else p.pid, p.name,
                                        _num(p.used_mb, 0))
             for g in snap.gpus for p in g.processes]
    out += ["", "## 进程", ""]
    out += (["| GPU | PID | 进程 | 显存 MiB |", "|---|---|---|---|"] + procs
            if procs else ["无运行进程。"])
    out += ["", "## 告警", ""]
    out += (["- ⚠️ **%s >= %g** — %s" % (a.kind, a.threshold, a.message) for a in snap.alerts]
            or ["无告警（当前阈值下全部指标正常）。"])
    pl = snap.placement
    if pl is not None:
        out += ["", "## 分配建议", "",
                "- 需求: %.2f GiB（%d MiB），安全水位 %g"
                % (pl["need_gb"], pl["need_mb"], pl["watermark"]),
                "- 推荐卡: %s" % ("#" + str(pl["recommended"])
                                if pl["recommended"] is not None else "无"), "",
                "| 建议 | idx | 名称 | 可分配 MiB | 本任务后占用 | 结论 |",
                "|---|---|---|---|---|---|"]
        for c in pl["candidates"]:
            out.append("| %s | %s | %s | %s | %s | %s |" % (
                "#" + str(c["rank"]) if c.get("rank") else "-", c["index"], c["name"],
                _num(c["avail_mb"], 0),
                "n/a" if c["after_used_pct"] is None else "%.1f%%" % c["after_used_pct"],
                "✅ FIT" if c["fit"] else "❌ NO (%s)" % c["reason"]))
    if snap.warnings:
        out += ["", "## 采集告警（warnings）", ""] + ["- " + w for w in snap.warnings]
    return "\n".join(out) + "\n"

def to_json_dict(snap: Snapshot) -> Dict[str, object]:
    def gpu(g: Gpu) -> Dict[str, object]:
        return {"index": g.index, "name": g.name, "uuid": g.uuid,
                "vram_total_mb": g.vram_total_mb, "vram_used_mb": g.vram_used_mb,
                "vram_free_mb": g.vram_free_mb, "util_percent": g.util_percent,
                "temp_c": g.temp_c, "power_draw_w": g.power_draw_w,
                "power_limit_w": g.power_limit_w,
                "processes": [{"pid": p.pid, "name": p.name, "used_mb": p.used_mb}
                              for p in g.processes],
                "alerts": [_alert_dict(a) for a in g.alerts]}
    return {"schema_version": snap.schema_version, "mode": snap.mode, "host": snap.host,
            "collected_at": snap.collected_at, "nvidia_smi": snap.nvidia_smi,
            "driver_version": snap.driver_version,
            "thresholds": {"vram_warn": snap.thresholds.vram_warn,
                           "temp_warn": snap.thresholds.temp_warn,
                           "util_warn": snap.thresholds.util_warn},
            "gpus": [gpu(g) for g in snap.gpus],
            "alerts": [_alert_dict(a) for a in snap.alerts],
            "warnings": list(snap.warnings), "placement": snap.placement}

def safe_host(host: str) -> str:
    """文件名净化，防路径穿越（第二层防线，第一层是 host 白名单）。"""
    return re.sub(r"[^A-Za-z0-9._-]", "_", host or "unknown")

def validate_target(value: str, what: str) -> str:
    """host/user 白名单：拒空、拒 '-' 开头（防被 ssh 当选项）、拒白名单外字符。"""
    if not value:
        raise UsageError("%s 不能为空" % what, hints=("--help 查看用法",))
    if value.startswith("-"):
        raise UsageError("%s 不能以 '-' 开头：%r" % (what, value),
                         hints=("host/user 会被 ssh 当选项解析，故直接拒绝", "--help 查看用法"))
    if not _HOST_RE.fullmatch(value):
        raise UsageError("%s 含非法字符（仅允许 A-Z a-z 0-9 . _ -）：%r\n"
                         "多主机/跳板机请用 ~/.ssh/config 的 ProxyJump，见 README" % (what, value),
                         hints=("单主机名，例如 gpu01", "跳板机请配置 ProxyJump 后用别名",
                                "--help 查看用法"))
    return value

def resolve_nvidia_smi(cli_path: Optional[str]) -> str:
    """优先级：--nvidia-smi > NVIDIA_SMI_PATH > PATH 搜索；显式给错路径立即失败不回落。"""
    for cand in (cli_path, os.environ.get("NVIDIA_SMI_PATH")):
        if cand:
            if Path(cand).is_file():
                return str(Path(cand))
            raise CollectError("nvidia-smi 不可用：%s 不存在或不是文件" % cand, hints=HINTS_NO_SMI)
    found = shutil.which("nvidia-smi")
    if not found:
        raise CollectError("未找到 nvidia-smi", hints=HINTS_NO_SMI)
    return found

def run_cmd(argv: List[str], timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess:
    """纯 argv 列表调用：不经本地 shell、不 check，编码容错（errors=replace）。"""
    try:
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise CollectError("命令超时（%ds）：%s" % (timeout, argv[0]),
                           hints=("检查主机可达性与网络抖动", "换端口后重试", "手工 ssh 排查"))
    except OSError as exc:
        raise CollectError("命令无法执行：%s（%s）" % (argv[0], exc), hints=HINTS_NO_SMI)

def _tail(text: Optional[str], limit: int = 20) -> str:
    """远端 stderr 只留前 20 行，并剥 ANSI 与除换行/制表外的控制字符（防转义注入）。"""
    if not text:
        return ""
    clean = "".join(ch for ch in _ANSI_RE.sub("", text) if ch in "\n\t" or ord(ch) >= 32)
    return "\n".join(clean.splitlines()[:limit])

def _new_snapshot(mode: str, host: str, smi: Optional[str], text: str, fields: List[str],
                  gpus: List[Gpu], warnings: List[str]) -> Snapshot:
    return Snapshot(SCHEMA_VERSION, mode, host, datetime.now().strftime(TS_FORMAT), smi,
                    parse_driver_version(text, fields), gpus, [], warnings,
                    Thresholds(DEFAULTS["vram_warn"], DEFAULTS["temp_warn"],
                               DEFAULTS["util_warn"]))

def _query_gpu(smi: str, warnings: List[str]) -> Tuple[str, List[str]]:
    """带 L1/L2 容错阶梯的主查询：字段不受支持则剔除重试（最多 2 次），再退最小集。"""
    fields = list(FIELDS_CORE)
    for attempt in range(3):
        cp = run_cmd([smi, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
        if cp.returncode == 0:
            if cp.stdout.strip():
                return cp.stdout, fields
            raise CollectError("nvidia-smi 无输出（可能无 GPU、驱动异常或 No devices found）",
                               hints=HINTS_NO_SMI)
        m = _FIELD_ERR_RE.search(cp.stderr or "")
        if not m:
            raise CollectError("nvidia-smi 执行失败 rc=%d：%s"
                               % (cp.returncode, _tail(cp.stderr) or "无 stderr"),
                               hints=HINTS_NO_SMI)
        bad = m.group(1)
        if bad in fields and len(fields) > len(FIELDS_MIN):
            fields = [f for f in fields if f != bad]
            warnings.append("驱动不支持字段 %s，已剔除后重试（第 %d 次）" % (bad, attempt + 1))
            continue
        break
    warnings.append("多次降级仍失败，已降级到最小字段集 index,name,memory.total,memory.used")
    cp = run_cmd([smi, "--query-gpu=" + ",".join(FIELDS_MIN), "--format=csv,noheader,nounits"])
    if cp.returncode == 0 and cp.stdout.strip():
        return cp.stdout, list(FIELDS_MIN)
    raise CollectError("最小字段集仍失败 rc=%d：%s"
                       % (cp.returncode, _tail(cp.stderr) or "无 stderr"), hints=HINTS_NO_SMI)

def _query_apps(smi: str, warnings: List[str]) -> Dict[str, List[Proc]]:
    """进程表独立一次查询，失败只记 warning，不影响主表（L3）。"""
    try:
        cp = run_cmd([smi, "--query-compute-apps=" + ",".join(FIELDS_APPS),
                      "--format=csv,noheader,nounits"])
    except CollectError as exc:
        warnings.append("进程表不可用（%s），进程列留空" % exc.message)
        return {}
    if cp.returncode != 0:
        warnings.append("进程表不可用（rc=%d），进程列留空" % cp.returncode)
        return {}
    if not cp.stdout.strip():
        warnings.append("进程表为空（No running processes found），进程列留空")
        return {}
    return parse_compute_apps(cp.stdout)

def collect_local(smi: str) -> Snapshot:
    warnings: List[str] = []
    text, fields = _query_gpu(smi, warnings)
    gpus = parse_smi_csv(text, fields, warnings)
    if not gpus:
        raise CollectError("解析后未得到任何 GPU 记录（输出为空或字段全不符）", hints=HINTS_NO_SMI)
    attach_processes(gpus, _query_apps(smi, warnings))
    return _new_snapshot("local", "localhost", smi, text, fields, gpus, warnings)

def build_ssh_argv(host: str, user: Optional[str], port: int, key: Optional[str],
                   remote_cmd: str) -> List[str]:
    """纯 list argv；remote_cmd 必须是常量，用户输入永不进入命令行。"""
    dest = ("%s@%s" % (user, host)) if user else host   # 两段均已过白名单，拼接安全
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
            "-p", str(int(port))]
    if key:
        argv += ["-i", key]          # 只当路径传，不读内容、不打印
    return argv + ["--", dest, remote_cmd]

def collect_remote(host: str, user: Optional[str], port: int, key: Optional[str]) -> Snapshot:
    validate_target(host, "--host")
    if user:
        validate_target(user, "--ssh-user")
    cp = run_cmd(build_ssh_argv(host, user, port, key, SMI_REMOTE_CMD))
    if cp.returncode != 0:
        raise CollectError("远端采集失败 rc=%d host=%s\n%s"
                           % (cp.returncode, host, _tail(cp.stderr) or "无 stderr"),
                           hints=HINTS_REMOTE)
    if not cp.stdout.strip():
        raise CollectError("远端 nvidia-smi 无输出 host=%s" % host, hints=HINTS_REMOTE)
    warnings: List[str] = []
    gpus = parse_smi_csv(cp.stdout, FIELDS_CORE, warnings)
    if not gpus:
        raise CollectError("远端输出解析后无 GPU 记录 host=%s" % host, hints=HINTS_REMOTE)
    return _new_snapshot("remote", host, "nvidia-smi", cp.stdout, FIELDS_CORE, gpus, warnings)

def collect_mock() -> Snapshot:
    warnings: List[str] = []
    gpus = parse_smi_csv(MOCK_SMI_TEXT, FIELDS_CORE, warnings)
    attach_processes(gpus, parse_compute_apps(MOCK_APPS_TEXT))
    return _new_snapshot("mock", "localhost", "<mock>", MOCK_SMI_TEXT, FIELDS_CORE, gpus,
                         warnings)

def collect_csv_file(path: str) -> Snapshot:
    """--csv-file：读取一份 nvidia-smi 同构 CSV 做解析测试，走与三模式相同的链路。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CollectError("无法读取 CSV 文件：%s（%s）" % (path, exc))
    warnings: List[str] = []
    gpus = parse_smi_csv(text, FIELDS_CORE, warnings)
    if not gpus:
        raise CollectError("CSV 文件解析后未得到任何 GPU 记录：%s" % path)
    return _new_snapshot("csv", "localhost", path, text, FIELDS_CORE, gpus, warnings)

def collect(args: argparse.Namespace) -> Snapshot:
    """唯一分发点：三模式都只产出 CSV 文本，之后走同一条解析/判定/渲染链路。"""
    if args.mock:
        snap = collect_mock()
    elif args.csv_file:
        snap = collect_csv_file(args.csv_file)
    elif args.host:
        snap = collect_remote(args.host, args.ssh_user, args.ssh_port, args.ssh_key)
    else:
        snap = collect_local(resolve_nvidia_smi(args.nvidia_smi))
    snap.thresholds = Thresholds(args.vram_warn, args.temp_warn, args.util_warn)
    snap.alerts = evaluate_alerts(snap.gpus, snap.thresholds)
    by_index = {g.index: g for g in snap.gpus}
    for alert in snap.alerts:
        by_index[alert.gpu_index].alerts.append(alert)
    if args.need_gb is not None:
        snap.placement = build_plan(snap.gpus, args.need_gb, args.watermark, snap.thresholds)
        if snap.placement["recommended"] is None:
            snap.warnings.append(
                "无卡可容纳 %.2f GiB（水位 %g）：建议降 batch/seq_len 或换精度，"
                "并用 src/vram_est.py 重算后调 --watermark" % (args.need_gb, args.watermark))
    return snap

def write_outputs(md: str, js: str, out_dir: str, host: str, ts: str) -> Tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = "gpu_snapshot_%s_%s" % (safe_host(host), ts)
    paths = (target / (stem + ".md"), target / (stem + ".json"))
    for path, body in zip(paths, (md, js)):
        path.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
    return paths

def render_error(title: str, message: str, hints: Tuple[str, ...]) -> str:
    lines = ["[%s] %s" % (title, message)]
    return "\n".join(lines + ["  %d) %s" % (i, h) for i, h in enumerate(hints, 1)]) + "\n"

class _Parser(argparse.ArgumentParser):
    """把 argparse 的 exit(2) 统一成 UsageError，保证参数错误恒为退出码 3。"""

    def error(self, message: str):  # type: ignore[override]
        raise UsageError(message, hints=("python src/gpu_snapshot.py --help",))

def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="gpu_snapshot.py",
        description="GPU 快照：三模式采集 -> 解析 -> 告警 -> 分配建议 -> markdown/JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("退出码：0 正常 / 1 有告警 / 2 采集失败 / 3 参数错误（优先级 3>2>1>0）\n"
                "示例：\n  python src/gpu_snapshot.py --mock\n"
                "  python src/gpu_snapshot.py --mock --json --need-gb 20\n"
                "  python src/gpu_snapshot.py --host gpu01 --ssh-user ubuntu\n"
                "告警阈值为 0 表示关闭该项；分配水位默认 " + str(WATERMARK)
                + "；建议采样间隔 >= 5s（本工具是快照，不做去抖）。\n"))
    src = p.add_mutually_exclusive_group()
    src.add_argument("--mock", action="store_true", help="用内置 CSV 文本走完整链路（无卡演示）")
    src.add_argument("--csv-file", metavar="PATH",
                     help="从 nvidia-smi 同构 CSV 文件读取并解析（解析测试入口）")
    src.add_argument("--host", metavar="HOST",
                     help="远端单主机名（多主机/跳板机用 ~/.ssh/config 的 ProxyJump）")
    p.add_argument("--json", action="store_true", help="stdout 输出 JSON（默认 markdown）")
    p.add_argument("--vram-warn", type=float, default=DEFAULTS["vram_warn"], metavar="PCT",
                   help="显存占用率告警阈值%%，默认 " + str(DEFAULTS["vram_warn"]) + "，0 = 关闭")
    p.add_argument("--temp-warn", type=float, default=DEFAULTS["temp_warn"], metavar="C",
                   help="温度告警阈值摄氏度，默认 " + str(DEFAULTS["temp_warn"]) + "，0 = 关闭")
    p.add_argument("--util-warn", type=float, default=DEFAULTS["util_warn"], metavar="PCT",
                   help="利用率告警阈值%%，默认 " + str(DEFAULTS["util_warn"]) + "，0 = 关闭")
    p.add_argument("--nvidia-smi", metavar="PATH", help="显式指定 nvidia-smi 路径")
    p.add_argument("--ssh-user", metavar="USER", help="ssh 用户名（白名单校验）")
    p.add_argument("--ssh-port", type=int, default=22, metavar="PORT", help="ssh 端口，默认 22")
    p.add_argument("--ssh-key", metavar="PATH", help="ssh 私钥路径（只作 -i 传参，不读内容）")
    p.add_argument("--need-gb", type=float, metavar="GIB",
                   help="本任务所需显存（GiB），取自 vram_est.py 的 total_gb")
    p.add_argument("--watermark", type=float, default=WATERMARK, metavar="R",
                   help="分配安全水位 (0,1]，默认 " + str(WATERMARK))
    p.add_argument("--out-dir", metavar="DIR", help="落盘目录：同时写 .md 与 .json")
    return p

def _validate_args(args: argparse.Namespace) -> None:
    for name in ("vram_warn", "temp_warn", "util_warn"):
        if getattr(args, name) < SENTINEL_OFF:
            raise UsageError("--%s 不能为负：%s" % (name.replace("_", "-"), getattr(args, name)),
                             hints=("传 0 表示关闭该项", "--help 查看用法"))
    if args.need_gb is not None and args.need_gb <= 0:
        raise UsageError("--need-gb 必须大于 0：%s" % args.need_gb,
                         hints=("先跑 src/vram_est.py --json 取 total_gb", "--help 查看用法"))
    if not 0.0 < args.watermark <= 1.0:
        raise UsageError("--watermark 必须在 (0,1] 区间：%s" % args.watermark,
                         hints=("默认 " + str(WATERMARK) + " 预留 10% 给峰值与碎片",
                                "--help 查看用法"))
    if args.ssh_port is not None and not 1 <= args.ssh_port <= 65535:
        raise UsageError("--ssh-port 必须在 1..65535：%s" % args.ssh_port,
                         hints=("默认 22", "--help 查看用法"))

def main(argv: Optional[List[str]] = None) -> int:
    """唯一出口：所有路径都返回退出码，不抛 traceback。"""
    try:
        args = build_parser().parse_args(argv)
        _validate_args(args)
        snap = collect(args)
        md = render_markdown(snap)
        payload = json.dumps(to_json_dict(snap), ensure_ascii=False, indent=2)
        text = payload if args.json else md
        if not text.endswith("\n"):
            text += "\n"
        if args.out_dir:
            write_outputs(md, payload, args.out_dir, snap.host, snap.collected_at)
        sys.stdout.write(text)
        sys.stdout.flush()
        return EXIT_ALERTS if snap.alerts else EXIT_OK
    except UsageError as exc:
        sys.stderr.write(render_error("参数错误", exc.message, exc.hints))
        return EXIT_USAGE
    except CollectError as exc:
        sys.stderr.write(render_error("采集失败", exc.message, exc.hints))
        return EXIT_COLLECT
    except SystemExit as exc:                 # --help / argparse 主动退出
        return int(exc.code or EXIT_OK)
    except KeyboardInterrupt:
        sys.stderr.write(render_error("已中断", "用户中断", ("重跑命令", "--help 查看用法")))
        return EXIT_COLLECT
    except Exception as exc:                  # 兜底：不打印 traceback（G4）
        sys.stderr.write(render_error("内部错误", "%s: %s" % (type(exc).__name__, exc),
                                      ("用 --mock 复现最小用例", "检查 --out-dir 是否可写",
                                       "提交 issue 时附完整命令")))
        return EXIT_COLLECT

if __name__ == "__main__":
    raise SystemExit(main())
