#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gateway_monitor.py —— deepseek-gateway-monitor 纯只读网关监控台（单文件 CLI）

- 零依赖：仅 Python 标准库（urllib / argparse / concurrent.futures / json / math / threading / socket / time / os）
- 三个子命令：
    health  探测网关可达性 + 延迟（GET /v1/models 回退 /health）
    bench   并发压测（限流默认开，纯只读 GET），输出 p50/p95/p99 + 吞吐 + 失败率
    usage   token 用量日报（自动识别 vLLM 后端时取 /metrics 真实值；否则优先 /v1/usage，失败则按压测样本/经验常数估算并标“估算”）
- 全局：--mock（内置样例全链路离线）、--json（机器可读）、--base-url、--header、--timeout
- vLLM 模式：自动探测后端（GET /metrics 命中 vllm: 前缀即判为 vLLM），命中后用量取真实 token 指标（prompt/generation/请求数/KV 缓存占用），健康额外带版本号与在跑/排队请求；非 vLLM 后端走原有逻辑，向后兼容
- 退出码：0 正常 / 1 告警或健康失败 / 2 采集失败 / 3 参数错误

dev 不变式（见 docs/04、docs/08）：
  A. 探测与解析解耦为纯函数（probe 仅返回三元组，parse_* / quantile 为纯函数）
  B. mock 仅切换「数据来源」，流经与真实响应完全相同的解析/统计函数
  C. header 只从 --header 命令行参数来，绝不读环境变量/配置文件密钥
  D. 所有请求 method 受限（仅 GET/HEAD），构造层强制白名单校验
"""

import argparse
import json
import math
import os
import re
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------- #
# 常量（硬约束，不可改）
# --------------------------------------------------------------------------- #
DEFAULT_BASE = "http://192.168.112.252:8080"   # PRD 硬约束，不可改
DEFAULT_TIMEOUT = 5.0
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 16                            # 并发封顶 16（不变式 1）
DEFAULT_REQUESTS = 100
DEFAULT_HEALTH_COUNT = 3
READONLY_METHODS = ("GET", "HEAD")              # 不变式 D：method 白名单
AVG_PROMPT_TOKENS = 32                          # 估算用经验常数（假设值，输出声明）
AVG_COMPLETION_TOKENS = 8                       # 估算用经验常数（假设值，输出声明）
BENCH_SAMPLE_PATH = os.path.join(tempfile.gettempdir(), "gateway_monitor_bench.json")

# --------------------------------------------------------------------------- #
# 内置样例数据（--mock，字段与真实响应对齐；仅切换数据来源，不变式 B）
# --------------------------------------------------------------------------- #
MOCK_HEALTH = {
    "/health": {
        "status_code": 200,
        "latency_ms": 23.4,
        "body": {"status": "healthy", "uptime_s": 86400, "version": "v1"},
    },
    "/v1/models": {
        "status_code": 200,
        "latency_ms": 31.1,
        "body": {
            "object": "list",
            "data": [{"id": "deepseek-chat", "object": "model", "owned_by": "gateway"}],
        },
    },
}

# 压测延迟样本序列（100 个，均值≈30ms，最近秩法 p50≈27 / p95≈44 / p99≈52）
_BASE_LAT = [22, 25, 24, 28, 30, 27, 35, 41, 26, 23, 29, 33, 38, 45, 24, 26, 31, 27, 36, 42]
MOCK_BENCH_LATENCIES = [
    float(_BASE_LAT[i % len(_BASE_LAT)] + (i // len(_BASE_LAT)) * 2) for i in range(100)
]

# 用量样本（mock 源，非估算、非真实网关）
MOCK_USAGE = {
    "source": "mock",
    "date": "2026-10-28",
    "prompt_tokens": 128000,
    "completion_tokens": 86000,
    "total_tokens": 214000,
    "request_count": 100,
    "estimated": False,
}

# vLLM 模式 mock 数据（字段与 vLLM 真实 /metrics（Prometheus 文本）逐行对齐；仅切换数据来源，不变式 B）
MOCK_VERSION = {"version": "0.6.3"}
MOCK_METRICS_TEXT = """# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 128000
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 86000
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{status="success"} 95
vllm:request_success_total{status="failed"} 5
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
# HELP vllm:num_requests_waiting Number of requests currently waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 1
# HELP vllm:gpu_cache_usage_sys GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_sys gauge
vllm:gpu_cache_usage_sys 0.42
"""

_mock_bench_idx = {"i": 0}
_mock_bench_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# 不变式 D：只读 method 白名单校验
# --------------------------------------------------------------------------- #
def assert_readonly_method(method):
    """强制只允许 GET/HEAD，传入其它 method 在构造层即抛 ValueError。"""
    if method not in READONLY_METHODS:
        raise ValueError("只读护栏：拒绝非白名单请求方法 %r（仅允许 GET/HEAD）" % method)


# --------------------------------------------------------------------------- #
# 探测层（不变式 A/C/D）：仅负责发请求，返回归一化响应三元组
# --------------------------------------------------------------------------- #
def probe(url, headers=None, timeout=DEFAULT_TIMEOUT, mock=False, mock_kind="health"):
    """
    纯只读 GET 探测。返回 dict：
        status_code, body(解析后的 dict 或 None), raw_text(原始响应文本), latency_ms, error, error_type
    error_type ∈ {None, 'UNREACHABLE', 'AUTH', 'GATEWAY', 'HTTP_<code>'}
    绝不裸 raise 到顶层（不变式全局兜底另行处理）。
    注：raw_text 同时保留原始文本，供 /metrics（Prometheus 文本）等非 JSON 响应解析。
    """
    if mock:
        return _mock_probe(url, mock_kind)

    assert_readonly_method("GET")  # 不变式 D：强制 GET，杜绝写方法
    headers = dict(headers or {})
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        # 只读护栏：urlopen 不传 data=（Request 构造时也不允许），写方法无入口
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            raw = resp.read()
        lat = (time.perf_counter() - t0) * 1000
        raw_text = raw.decode("utf-8", "replace")
        body = _try_parse_json(raw)
        return {"status_code": status, "body": body, "raw_text": raw_text, "latency_ms": lat,
                "error": None, "error_type": None}
    except urllib.error.HTTPError as e:
        lat = (time.perf_counter() - t0) * 1000
        if e.code == 401:
            etype = "AUTH"
        elif e.code >= 500:
            etype = "GATEWAY"
        else:
            etype = "HTTP_%d" % e.code
        return {"status_code": e.code, "body": None, "raw_text": None, "latency_ms": lat,
                "error": str(e), "error_type": etype}
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        lat = (time.perf_counter() - t0) * 1000
        return {"status_code": None, "body": None, "raw_text": None, "latency_ms": lat,
                "error": str(e), "error_type": "UNREACHABLE"}


def _try_parse_json(raw):
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None


def _mock_probe(url, kind):
    """mock 数据来源：返回与真实响应结构完全一致的归一化 dict（零 socket）。"""
    if kind == "bench":
        with _mock_bench_lock:
            i = _mock_bench_idx["i"]
            _mock_bench_idx["i"] = (i + 1) % len(MOCK_BENCH_LATENCIES)
        lat = MOCK_BENCH_LATENCIES[i]
        return {
            "status_code": 200,
            "body": {"object": "list", "data": [{"id": "deepseek-chat", "object": "model", "owned_by": "gateway"}]},
            "raw_text": None,
            "latency_ms": lat, "error": None, "error_type": None,
        }
    if kind == "usage":
        # 字段与真实 /v1/usage 响应对齐，流经 _extract_usage_from_resp（不变式 B）
        return {"status_code": 200, "body": dict(MOCK_USAGE), "raw_text": None,
                "latency_ms": 0.0, "error": None, "error_type": None}
    if kind == "metrics":
        # vLLM /metrics（Prometheus 文本），含 vllm: 前缀，流经 parse_vllm_metrics（不变式 B）
        return {"status_code": 200, "body": None, "raw_text": MOCK_METRICS_TEXT,
                "latency_ms": 0.0, "error": None, "error_type": None}
    if kind == "version":
        return {"status_code": 200, "body": dict(MOCK_VERSION),
                "raw_text": json.dumps(MOCK_VERSION), "latency_ms": 0.0,
                "error": None, "error_type": None}
    if kind == "server_info":
        return {"status_code": 200, "body": {"vllm_config": {}, "vllm_env": {}},
                "raw_text": json.dumps({"vllm_config": {}, "vllm_env": {}}),
                "latency_ms": 0.0, "error": None, "error_type": None}
    # health / usage(非 bench) 走健康样例
    if "models" in url:
        s = MOCK_HEALTH["/v1/models"]
    else:
        s = MOCK_HEALTH["/health"]
    return {"status_code": s["status_code"], "body": s["body"], "raw_text": None,
            "latency_ms": s["latency_ms"], "error": None, "error_type": None}


# --------------------------------------------------------------------------- #
# 解析层纯函数（不变式 A）
# --------------------------------------------------------------------------- #
def parse_models(resp):
    """从 /v1/models 响应解析模型列表；返回 (ok, [model_ids])。"""
    body = resp.get("body")
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list) and len(data) > 0:
            ids = [m.get("id") for m in data if isinstance(m, dict)]
            if ids:
                return True, ids
    return False, []


def parse_health(resp):
    """从 /health 响应解析健康状态；返回 bool。"""
    body = resp.get("body")
    if isinstance(body, dict):
        st = body.get("status")
        if st in ("ok", "healthy"):
            return True
    return False


def is_reachable(resp):
    """可达判定：2xx 且（模型列表非空 或 健康状态 ok）。"""
    sc = resp.get("status_code")
    if not (isinstance(sc, int) and 200 <= sc < 300):
        return False
    return parse_models(resp)[0] or parse_health(resp)


# --------------------------------------------------------------------------- #
# 统计层纯函数（不变式 A/2/4/5）
# --------------------------------------------------------------------------- #
def quantile(sorted_latencies, p):
    """
    最近秩法分位（确定性、零依赖）：
        idx = ceil(p/100 * n) - 1，边界夹紧；n==0 返回 None。
    """
    n = len(sorted_latencies)
    if n == 0:
        return None
    idx = math.ceil(p / 100.0 * n) - 1
    idx = max(0, min(n - 1, idx))
    return sorted_latencies[idx]


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def parse_headers(header_list):
    """--header "Key: Value" 可多次，拼接为 dict（不变式 C：仅来自参数）。"""
    headers = {}
    for h in (header_list or []):
        if ":" not in h:
            # 仍按原样尝试，但不抛异常，仅跳过无效项
            continue
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def join_url(base, endpoint):
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return base.rstrip("/") + "/" + endpoint.lstrip("/")


def fmt_ms(v):
    return "%.2f" % v if v is not None else "n/a"


# --------------------------------------------------------------------------- #
# 子命令：health
# --------------------------------------------------------------------------- #
def cmd_health(args, mock):
    headers = parse_headers(args.header)
    timeout = args.timeout
    endpoints = [args.endpoint] if args.endpoint else ["/v1/models", "/health"]
    count = args.count

    per_ep = []
    any_reachable = False
    error_summary = {}
    all_latencies = []

    for ep in endpoints:
        ok_lat = []
        ok_rounds = 0
        last_err = None
        models = []
        for _ in range(count):
            url = join_url(args.base_url, ep)
            resp = probe(url, headers, timeout, mock, mock_kind="health")
            if is_reachable(resp):
                ok_rounds += 1
                ok_lat.append(resp["latency_ms"])
                any_reachable = True
                if not models:
                    models = parse_models(resp)[1]
            else:
                et = resp["error_type"] or ("HTTP_%s" % resp["status_code"])
                last_err = et
                error_summary[et] = error_summary.get(et, 0) + 1
        avg = (sum(ok_lat) / len(ok_lat)) if ok_lat else None
        all_latencies.extend(ok_lat)
        per_ep.append({
            "endpoint": ep,
            "reachable_rounds": ok_rounds,
            "total_rounds": count,
            "avg_latency_ms": avg,
            "models": models,
            "last_error": (None if ok_rounds == count else last_err),
        })

    overall_avg = (sum(all_latencies) / len(all_latencies)) if all_latencies else None

    # 后端识别（best-effort，识别失败不影响健康判定）
    backend = {"kind": "unknown", "version": None}
    vllm_info = None
    try:
        backend = detect_backend(args.base_url, headers, timeout, mock)
        if backend["kind"] == "vllm" and backend.get("metrics_text"):
            m = parse_vllm_metrics(backend["metrics_text"])
            vllm_info = {
                "gpu_cache_usage": m["gpu_cache_usage"],
                "requests_running": m["requests_running"],
                "requests_waiting": m["requests_waiting"],
            }
    except Exception:
        backend = {"kind": "unknown", "version": None}

    result = {
        "command": "health",
        "mock": mock,
        "reachable": any_reachable,
        "backend": backend["kind"],
        "backend_version": backend.get("version"),
        "vllm": vllm_info,
        "overall_avg_latency_ms": overall_avg,
        "endpoints": per_ep,
        "error_summary": error_summary,
    }
    if not any_reachable:
        result["degraded"] = True
        result["message"] = "网关不可达（UNREACHABLE / 鉴权失败 AUTH / 网关错误 GATEWAY），请检查网络或尝试 --header"
        emit(result, args)
        return 1  # 健康失败

    result["message"] = "网关可达"
    emit(result, args)
    return 0


# --------------------------------------------------------------------------- #
# 子命令：bench（并发压测，限流默认开，纯只读 GET）
# --------------------------------------------------------------------------- #
def cmd_bench(args, mock):
    concurrency = min(MAX_CONCURRENCY, max(1, args.concurrency))  # 封顶 16
    requests = max(1, args.requests)
    timeout = args.timeout
    headers = parse_headers(args.header)
    rate = args.rate  # 可选令牌桶，默认 None（不限速，仅靠信号量兜底）

    if args.dry_run:
        endpoints = [args.endpoint] if args.endpoint else ["/health", "/v1/models"]
        result = {
            "command": "bench",
            "mock": mock,
            "dry_run": True,
            "concurrency": concurrency,
            "requests": requests,
            "success_count": 0,
            "failed_count": 0,
            "success_rate": 0.0,
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "throughput_rps": 0.0,
            "wall_clock_s": 0.0,
            "error_summary": {},
            "degraded": False,
            "message": "压测计划（dry-run，未发请求）",
            "plan": {
                "workers": concurrency,
                "concurrency_cap": MAX_CONCURRENCY,
                "rate_limit": rate,
                "endpoints": endpoints,
                "estimated_qps": (min(concurrency / 0.03, rate) if rate else concurrency / 0.03),
            },
        }
        emit(result, args)
        return 0

    sem = threading.Semaphore(concurrency)            # 不变式 1：硬控并发默认开且可关
    counter = {"i": 0}
    counter_lock = threading.Lock()
    results = []
    results_lock = threading.Lock()
    endpoints = [args.endpoint] if args.endpoint else ["/health", "/v1/models"]

    # 令牌桶（可选）
    bucket = None
    if rate:
        bucket = {"tokens": float(concurrency), "last": time.perf_counter(),
                  "capacity": float(concurrency), "rate": float(rate)}

    def acquire_token():
        if not bucket:
            return
        now = time.perf_counter()
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["rate"])
        bucket["last"] = now
        if bucket["tokens"] < 1.0:
            time.sleep((1.0 - bucket["tokens"]) / bucket["rate"])
        bucket["tokens"] -= 1.0

    def worker():
        while True:
            with counter_lock:
                if counter["i"] >= requests:
                    return
                counter["i"] += 1
            sem.acquire()
            try:
                acquire_token()
                ep = endpoints[counter["i"] % len(endpoints)]
                url = join_url(args.base_url, ep)
                resp = probe(url, headers, timeout, mock, mock_kind="bench")
                lat = resp["latency_ms"]  # 使用 probe 实测/样例延迟，保证 mock 样本进入统计
                ok = is_reachable(resp)
                with results_lock:
                    results.append({
                        "ok": ok,
                        "latency_ms": lat if ok else None,  # 不变式 4：仅成功样本入延迟
                        "error_type": (None if ok else (resp["error_type"] or "HTTP_%s" % resp["status_code"])),
                    })
            finally:
                sem.release()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        ex.map(lambda _: worker(), range(concurrency))
    end = time.perf_counter()

    success = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    success_count = len(success)
    latencies = sorted(r["latency_ms"] for r in success)  # 已保证非 None（不变式 4）
    wall = (end - start)

    # mock 下无真实网络，用确定性挂钟：sum(lat_ms)/1000/concurrency，避免吞吐虚高
    if mock and latencies:
        wall = sum(latencies) / 1000.0 / float(concurrency)

    p50 = quantile(latencies, 50)
    p95 = quantile(latencies, 95)
    p99 = quantile(latencies, 99)
    throughput = (success_count / wall) if wall > 0 else 0.0
    success_rate = (success_count / requests) if requests else 0.0

    # 错误分级汇总
    err_summary = {}
    for r in failed:
        et = r["error_type"] or "UNKNOWN"
        err_summary[et] = err_summary.get(et, 0) + 1

    degraded = (success_count > 0 and success_count < requests)  # R3 部分失败降级
    all_failed = (success_count == 0)

    # 落盘压测样本供 usage --from-bench 复用（仅本地，无网关写回）
    try:
        with open(BENCH_SAMPLE_PATH, "w", encoding="utf-8") as f:
            json.dump({"success_count": success_count, "requests": requests,
                       "p50": p50, "p95": p95, "p99": p99,
                       "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
                       "ts": time.time()}, f)
    except Exception:
        pass

    result = {
        "command": "bench",
        "mock": mock,
        "concurrency": concurrency,
        "requests": requests,
        "success_count": success_count,
        "failed_count": len(failed),
        "success_rate": round(success_rate, 4),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_rps": round(throughput, 4),
        "wall_clock_s": round(wall, 4),
        "error_summary": err_summary,
        "degraded": degraded,
    }
    if all_failed:
        result["message"] = "全部请求失败（压测告警）"
        emit(result, args)
        return 1  # 告警：全失败
    if degraded:
        result["message"] = "部分请求失败，已降级完成（成功率 < 100%%）"
    else:
        result["message"] = "压测完成"
    emit(result, args)
    return 0  # 部分失败/全部成功均退出 0（R3 降级语义）


# --------------------------------------------------------------------------- #
# 子命令：usage（token 用量日报）
# --------------------------------------------------------------------------- #
def cmd_usage(args, mock):
    headers = parse_headers(args.header)
    timeout = args.timeout
    date = args.date or time.strftime("%Y-%m-%d")

    # 0) 后端识别：vLLM 原生暴露 /metrics 真实 token 指标，优先取真实值（绝不等同估算）
    backend = detect_backend(args.base_url, headers, timeout, mock)
    if backend["kind"] == "vllm" and backend.get("metrics_text"):
        m = parse_vllm_metrics(backend["metrics_text"])
        result = {
            "command": "usage",
            "mock": mock,
            "date": date,
            "backend": "vllm",
            "backend_version": backend.get("version"),
            "source": ("mock（内置样例，非真实网关）" if mock else "真实（来源 vLLM /metrics）"),
            "estimated": False,
            "prompt_tokens": m["prompt_tokens"],
            "completion_tokens": m["completion_tokens"],
            "total_tokens": m["total_tokens"],
            "request_count": m["request_count"],
            "gpu_cache_usage": m["gpu_cache_usage"],
            "requests_running": m["requests_running"],
            "requests_waiting": m["requests_waiting"],
        }
        emit(result, args)
        return 0

    # 1) 非 vLLM：探测 /v1/usage 真实接口（仅只读 GET；mock 流经 _extract_usage_from_resp，不变式 B）
    url = join_url(args.base_url, "/v1/usage")
    resp = probe(url, headers, timeout, mock, mock_kind="usage")
    real_usage = _extract_usage_from_resp(resp)

    if real_usage is not None:
        result = {
            "command": "usage",
            "mock": mock,
            "date": date,
            "source": ("mock（内置样例，非真实网关）" if mock else "真实（来源 /v1/usage）"),
            "estimated": False,
            "prompt_tokens": real_usage.get("prompt_tokens"),
            "completion_tokens": real_usage.get("completion_tokens"),
            "total_tokens": real_usage.get("total_tokens"),
            "request_count": real_usage.get("request_count"),
        }
        emit(result, args)
        return 0

    # 2) 降级估算（强制标注“估算”，不变式 3）
    estimate = None
    estimate_src = None
    try:
        if args.from_bench and os.path.exists(BENCH_SAMPLE_PATH):
            with open(BENCH_SAMPLE_PATH, "r", encoding="utf-8") as f:
                sample = json.load(f)
            success_requests = sample.get("success_count", 0)
            estimate_src = "压测样本（--from-bench，本地 %s）" % os.path.basename(BENCH_SAMPLE_PATH)
        else:
            success_requests = args.requests
            estimate_src = "经验常数（prompt≈%d, completion≈%d，无用量接口）" % (
                AVG_PROMPT_TOKENS, AVG_COMPLETION_TOKENS)

        if success_requests <= 0:
            success_requests = args.requests

        est_prompt = success_requests * AVG_PROMPT_TOKENS
        est_completion = success_requests * AVG_COMPLETION_TOKENS
        estimate = {
            "prompt_tokens": est_prompt,
            "completion_tokens": est_completion,
            "total_tokens": est_prompt + est_completion,
            "request_count": success_requests,
        }
    except Exception:
        estimate = None

    if estimate is None:
        result = {
            "command": "usage",
            "mock": mock,
            "date": date,
            "source": None,
            "estimated": None,
            "message": "用量采集失败：真实源不可得且估算不可用",
        }
        emit(result, args)
        return 2  # 采集失败

    result = {
        "command": "usage",
        "mock": mock,
        "date": date,
        "source": estimate_src,
        "estimated": True,
        "prompt_tokens": estimate["prompt_tokens"],
        "completion_tokens": estimate["completion_tokens"],
        "total_tokens": estimate["total_tokens"],
        "request_count": estimate["request_count"],
        "note": "**估算**（非真实网关数据，请勿直接用于计费）",
    }
    emit(result, args)
    return 0


def _extract_usage_from_resp(resp):
    """从 /v1/usage 响应或响应头提取用量；2xx 且含数值则返回 dict，否则 None。"""
    sc = resp.get("status_code")
    if not (isinstance(sc, int) and 200 <= sc < 300):
        return None
    body = resp.get("body")
    if isinstance(body, dict):
        pt = body.get("prompt_tokens") or body.get("promptTokens")
        ct = body.get("completion_tokens") or body.get("completionTokens")
        tt = body.get("total_tokens") or body.get("totalTokens")
        rc = body.get("request_count") or body.get("requestCount")
        if tt is not None or (pt is not None and ct is not None):
            return {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt if tt is not None else (pt + ct),
                "request_count": rc,
            }
    return None


# --------------------------------------------------------------------------- #
# vLLM 后端识别 + Prometheus 指标解析（纯函数，不变式 A；仅只读 GET /metrics）
# --------------------------------------------------------------------------- #
def detect_backend(base_url, headers, timeout, mock):
    """
    后端识别（best-effort，绝不因识别失败而中断主流程）：
      1) GET /metrics 命中 vllm: 前缀 → vLLM（最稳信号）
      2) GET /server_info 含 vllm_config → vLLM
    返回 dict: {kind: 'vllm'|'unknown', version, metrics_text}
    metrics_text 在识别为 vLLM 时直接复用，避免二次抓取。
    """
    resp = probe(join_url(base_url, "/metrics"), headers, timeout, mock, mock_kind="metrics")
    raw = resp.get("raw_text") or ""
    if "vllm:" in raw:
        return {"kind": "vllm", "version": _version_via_endpoint(base_url, headers, timeout, mock),
                "metrics_text": raw}
    si = probe(join_url(base_url, "/server_info"), headers, timeout, mock, mock_kind="server_info")
    body = si.get("body")
    if isinstance(body, dict) and "vllm_config" in body:
        return {"kind": "vllm", "version": _version_via_endpoint(base_url, headers, timeout, mock),
                "metrics_text": raw}
    return {"kind": "unknown", "version": None, "metrics_text": None}


def _version_via_endpoint(base_url, headers, timeout, mock):
    """从 /version 取版本号（vLLM 返回其自身版本）；不可得返回 None。"""
    resp = probe(join_url(base_url, "/version"), headers, timeout, mock, mock_kind="version")
    body = resp.get("body")
    if isinstance(body, dict):
        v = body.get("version") or body.get("vllm_version")
        if v:
            return str(v)
    return None


def _parse_prom_text(text):
    """解析 Prometheus 文本为 {full_key: numeric}，忽略注释/空行/非数值行。"""
    metrics = {}
    if not text:
        return metrics
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([+-]?[\d.eE+]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), m.group(3)
        try:
            num = float(val)
        except ValueError:
            continue
        key = name if not labels else "%s{%s}" % (name, labels)
        metrics[key] = num
    return metrics


def _sum_metric(metrics, base):
    """对同名（忽略标签）的所有样本求和；未找到返回 (None, False)。"""
    total = 0.0
    found = False
    for k, v in metrics.items():
        if k.split("{")[0] == base:
            total += v
            found = True
    return (total if found else None), found


def parse_vllm_metrics(metrics_text):
    """
    从 vLLM /metrics 文本提取真实用量与运行时快照（不变式 B：流经纯函数）。
    返回 dict（各字段可能 None）：prompt_tokens/completion_tokens/total_tokens/
    request_count/gpu_cache_usage/requests_running/requests_waiting。
    优先 prompt_tokens_total + generation_tokens_total；二者皆缺时回退 iteration_tokens_total_sum 作总计。
    """
    metrics = _parse_prom_text(metrics_text)
    prompt_t, _ = _sum_metric(metrics, "vllm:prompt_tokens_total")
    gen_t, _ = _sum_metric(metrics, "vllm:generation_tokens_total")
    if prompt_t is None and gen_t is None:
        iter_sum, _ = _sum_metric(metrics, "vllm:iteration_tokens_total_sum")
        total_tokens = iter_sum
    else:
        total_tokens = (prompt_t or 0) + (gen_t or 0)
    req_count, _ = _sum_metric(metrics, "vllm:request_success_total")
    if req_count is None:
        req_count, _ = _sum_metric(metrics, "vllm:request_total_count")
    gpu_cache, _ = _sum_metric(metrics, "vllm:gpu_cache_usage_sys")
    running, _ = _sum_metric(metrics, "vllm:num_requests_running")
    waiting, _ = _sum_metric(metrics, "vllm:num_requests_waiting")
    return {
        "prompt_tokens": prompt_t,
        "completion_tokens": gen_t,
        "total_tokens": total_tokens,
        "request_count": req_count,
        "gpu_cache_usage": gpu_cache,
        "requests_running": running,
        "requests_waiting": waiting,
    }


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def format_text(result):
    """构建人类可读文本（与 stdout 一致），供打印或 --out 落盘复用。"""
    lines = []
    cmd = result.get("command")
    lines.append("=" * 56)
    if cmd == "health":
        lines.append("[health] 网关健康探测")
        lines.append("  可达: %s" % ("是" if result["reachable"] else "否"))
        lines.append("  后端: %s%s" % (result.get("backend", "unknown"),
                                       (" (v%s)" % result["backend_version"]) if result.get("backend_version") else ""))
        vllm = result.get("vllm")
        if vllm:
            lines.append("  vLLM: GPU_KV缓存占用=%s  在跑=%s  排队=%s" % (
                fmt_ms(vllm.get("gpu_cache_usage")), vllm.get("requests_running"),
                vllm.get("requests_waiting")))
        lines.append("  整体平均延迟: %s ms" % fmt_ms(result["overall_avg_latency_ms"]))
        for ep in result["endpoints"]:
            lines.append("  - %s  可达轮次 %d/%d  平均延迟 %s ms  模型 %s" % (
                ep["endpoint"], ep["reachable_rounds"], ep["total_rounds"],
                fmt_ms(ep["avg_latency_ms"]), (",".join(ep["models"]) or "-")))
        if result["error_summary"]:
            lines.append("  错误分级: %s" % json.dumps(result["error_summary"], ensure_ascii=False))
        if result.get("mock"):
            lines.append("  [mock] 离线样例数据")
    elif cmd == "bench":
        lines.append("[bench] 并发压测（只读 GET）")
        lines.append("  并发: %d  总请求: %d  成功: %d  失败: %d" % (
            result["concurrency"], result["requests"],
            result["success_count"], result["failed_count"]))
        lines.append("  p50=%s ms  p95=%s ms  p99=%s ms" % (
            fmt_ms(result["p50_ms"]), fmt_ms(result["p95_ms"]), fmt_ms(result["p99_ms"])))
        lines.append("  吞吐: %.2f req/s  成功率: %.2f%%  挂钟: %.2fs" % (
            result["throughput_rps"], result["success_rate"] * 100, result["wall_clock_s"]))
        if result["error_summary"]:
            lines.append("  错误分级: %s" % json.dumps(result["error_summary"], ensure_ascii=False))
        if result.get("degraded"):
            lines.append("  [降级] 部分请求失败，按 R3 降级完成（退出 0）")
        if result.get("mock"):
            lines.append("  [mock] 离线样例数据")
    elif cmd == "usage":
        lines.append("[usage] token 用量日报")
        lines.append("  日期: %s" % result["date"])
        if result.get("estimated"):
            lines.append("  来源: %s" % result["source"])
            lines.append("  **估算** prompt=%s completion=%s total=%s (请求数 %s)" % (
                result["prompt_tokens"], result["completion_tokens"],
                result["total_tokens"], result["request_count"]))
            lines.append("  说明: %s" % result["note"])
        elif result.get("source"):
            lines.append("  来源: %s" % result["source"])
            if result.get("backend"):
                lines.append("  后端: %s%s" % (result["backend"],
                    (" (v%s)" % result["backend_version"]) if result.get("backend_version") else ""))
            lines.append("  prompt=%s completion=%s total=%s (请求数 %s)" % (
                result["prompt_tokens"], result["completion_tokens"],
                result["total_tokens"], result["request_count"]))
            if result.get("gpu_cache_usage") is not None:
                lines.append("  vLLM: GPU_KV缓存占用=%s  在跑=%s  排队=%s" % (
                    result.get("gpu_cache_usage"), result.get("requests_running"),
                    result.get("requests_waiting")))
        else:
            lines.append("  %s" % result.get("message", "采集失败"))
        if result.get("mock"):
            lines.append("  [mock] 离线样例数据（source=mock，非真实网关）")
    lines.append("=" * 56)
    if result.get("message") and cmd != "usage":
        lines.append(result["message"])
    return "\n".join(lines)


def emit(result, args):
    """输出：有 --out 落盘（--json 写 JSON，否则写可读文本），无 --out 才打印 stdout。"""
    if args.out:
        if args.json:
            payload = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            payload = format_text(result)
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(payload)
                if not payload.endswith("\n"):
                    f.write("\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("落盘失败（--out %s）: %s\n" % (args.out, e))
        return
    print(format_text(result))


# --------------------------------------------------------------------------- #
# CLI 解析与入口
# --------------------------------------------------------------------------- #
class _ArgParser(argparse.ArgumentParser):
    """重写 error 使参数错误退出码为 3（不变式：3 参数错误）。"""

    def error(self, message):
        sys.stderr.write("参数错误: %s\n" % message)
        sys.exit(3)


def build_global_parser():
    """全局参数解析器（add_help=False，使 --help 下沉到子命令层）。"""
    g = _ArgParser(add_help=False, prog="gateway_monitor.py",
                   description="纯只读网关监控台（标准库零依赖）")
    g.add_argument("--mock", action="store_true", help="离线模式：使用内置样例全链路可跑（零 socket）")
    g.add_argument("--json", action="store_true", help="机器可读 JSON 输出")
    g.add_argument("--base-url", default=DEFAULT_BASE, help="网关 base-url（默认 %s，PRD 硬约束不可改）" % DEFAULT_BASE)
    g.add_argument("--header", action="append", default=[], help="透传请求头，可多次：--header 'Authorization: Bearer xxx'")
    g.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="请求超时秒数（默认 %.1f）" % DEFAULT_TIMEOUT)
    return g


def build_sub_parser():
    """子命令解析器（不含全局项；全局项由两阶段解析先行抽离，支持任意位置）。
    子命令别名对齐 PRD §1/§3：health↔check、bench↔stress、usage↔report（同指向同一处理函数）。
    """
    p = _ArgParser(prog="gateway_monitor.py", description="纯只读网关监控台（标准库零依赖）")
    sub = p.add_subparsers(dest="command")

    # health / check（PRD §1a 别名 check）
    for name in ("health", "check"):
        h = sub.add_parser(name, help="探测网关可达性 + 延迟（GET /v1/models 回退 /health）")
        h.add_argument("--endpoint", default=None, help="覆盖探测端点（默认 /v1/models 回退 /health）")
        h.add_argument("--count", type=int, default=DEFAULT_HEALTH_COUNT, help="探测轮数（默认 %d）" % DEFAULT_HEALTH_COUNT)
        h.add_argument("--out", default=None, help="落盘路径（JSON/文本，仅本地只读产物，不写回网关）")

    # bench / stress（PRD §1b 别名 stress）
    for name in ("bench", "stress"):
        b = sub.add_parser(name, help="并发压测（只读 GET，限流默认开）")
        b.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="总请求数（默认 %d）" % DEFAULT_REQUESTS)
        b.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发数（默认 %d，封顶 %d）" % (DEFAULT_CONCURRENCY, MAX_CONCURRENCY))
        b.add_argument("--rate", type=float, default=None, help="可选 QPS 令牌桶限速（默认不限速）")
        b.add_argument("--endpoint", default=None, help="覆盖压测端点（默认 /health、/v1/models）")
        b.add_argument("--dry-run", action="store_true", help="仅打印压测计划，不发请求")
        b.add_argument("--allow-echo", action="store_true", help="（只读构建中忽略）echo 路径在本纯只读版本中禁用")
        b.add_argument("--out", default=None, help="落盘路径（JSON/文本，仅本地只读产物，不写回网关）")

    # usage / report（PRD §1c 别名 report）
    for name in ("usage", "report"):
        u = sub.add_parser(name, help="token 用量日报（优先探测用量接口，失败则估算并标「估算」）")
        u.add_argument("--date", default=None, help="日报日期 YYYY-MM-DD（默认今天）")
        u.add_argument("--from-bench", action="store_true", help="复用最近压测样本估算（本地临时 JSON）")
        u.add_argument("--out", default=None, help="落盘路径（JSON/文本，仅本地只读产物，不写回网关）")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 两阶段解析：先抽离全局参数（任意位置，避免子命令默认值覆盖），再解析子命令
    g_parser = build_global_parser()
    g_args, remaining = g_parser.parse_known_args(argv)
    sub_parser = build_sub_parser()
    args = sub_parser.parse_args(remaining)

    # 合并全局项到 args（子命令层不再重复定义，杜绝 argparse 子命令默认值陷阱）
    args.mock = g_args.mock
    args.json = g_args.json
    args.base_url = g_args.base_url
    args.header = g_args.header
    args.timeout = g_args.timeout

    if not args.command:
        sub_parser.print_help()
        return 0

    # 全局兜底：未预期异常 → 退出码 3（不暴露栈）
    try:
        if args.command in ("health", "check"):
            return cmd_health(args, args.mock)
        if args.command in ("bench", "stress"):
            return cmd_bench(args, args.mock)
        if args.command in ("usage", "report"):
            return cmd_usage(args, args.mock)
        sub_parser.print_help()
        return 0
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("致命异常（已收敛，退出码 3）: %s\n" % e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
