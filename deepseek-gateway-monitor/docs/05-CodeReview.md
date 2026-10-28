# Code Review —— src/gateway_monitor.py（reviewer）

## TL;DR
- 结论：**放行**（修复 1 处不变式违背 + 标注 1 处文档级冲突，遗留 4 条建议不阻塞）。
- 已修复：08-不变式 B（mock 不经 if 特判）在 `cmd_usage` 被违背，已让 mock 与真实路径统一流经 `_extract_usage_from_resp`。
- 需主管决策：`socket.timeout/URLError` 的错误类型，04 要求 `TIMEOUT`、08 要求 `UNREACHABLE`，两文档冲突，待裁定。
- 10/10 不变式功能层面满足；1 处逻辑 bug 已修，无新增阻塞问题。

---

## [已修复]

### B-1（08-推理方案 §4-B：mock 不经 if 特判）—— 逻辑 bug，已修复
- **问题**：`cmd_usage` 顶部存在 `if mock:` 硬编码提前返回（原 L467-480），直接拼装 `MOCK_USAGE` 字典并 `return`，**完全绕过** `_extract_usage_from_resp` 解析函数。这违反"mock 仅切换数据来源、必须流经与真实响应相同的解析/统计函数"的强约束，reviewer 一票否决项。
- **修复**：
  - `cmd_usage` 删除提前返回，统一调用 `probe(url, headers, timeout, mock, mock_kind="usage")` → `_extract_usage_from_resp(resp)`；`source` 字段按 `mock` 取值区分标注。
  - `_mock_probe` 新增 `kind == "usage"` 分支，返回 `status_code=200, body=dict(MOCK_USAGE)`，结构与真实 `/v1/usage` 响应一致，交由同一解析函数提取。
- **验证**：`usage --mock --json` 输出与原行为一致（`estimated=false`，`source` 标注 mock，数值 128000/86000/214000/100）；`py_compile` 通过。

---

## [需主管决策]

### A-5（04-算法设计 §2.1/不变式5）—— 文档级冲突，待裁定
- **现象**：04 明确要求"超时（`socket.timeout/URLError`）一律记 `TIMEOUT` 失败"；08 §1.1 则将"URLError/socket.timeout → UNREACHABLE"。当前 `probe` 将这两类归入 `error_type="UNREACHABLE"`。
- **影响**：功能正确——超时均被判为失败（`is_reachable=False`）、不进入延迟序列、计入成功率分母，与两文档一致；**仅错误标签取值分歧**。
- **建议**：以 04 不变式 5 为准记为 `TIMEOUT`（更能区分"不可达"与"超时"两种失败语义），但属设计级取舍，需主管在两文档间统一口径后再改。
- **现状**：未改，等裁定。

---

## [建议不阻塞]

### A-1 限流默认开可关 —— 满足，仅备注
`Semaphore(concurrency)` 恒生效、封顶 16；`--rate` 令牌桶默认关闭，无 `--no-limit` 等绕过入口，符合 04 不变式 1。备注：不变式表述"可关"指速率桶可关、并发护栏不可关，实现与文档一致。

### B-2（08 §3.2）`--allow-echo` 死标志 —— 建议删除
`build_sub_parser` 给 bench 定义了 `--allow-echo`，注释称"只读构建中忽略"，实际为 no-op。纯只读版本不应保留暗示写能力（echo 通常需 POST/payload）的死开关，易误导。建议删除该参数。

### B-3（08 §4-B）`cmd_bench` 挂钟 `if mock:` 特判 —— 可接受
`if mock and latencies: wall = sum(latencies)/1000.0/concurrency` 以确定性挂钟替代真实网络耗时，避免吞吐虚高。未绕过解析/统计函数，属测量层调整，已在注释说明，可接受；仅标注其仍属"对 mock 的 if 分支"。

### B-4 `parse_headers` 静默丢弃无效头 —— 建议告警
无 `:` 的 `--header` 项被 `continue` 静默跳过（L223-225），用户拼写错误时无反馈。建议向 stderr 输出一条告警便于排查。

### B-5 `quantile` 不校验排序 —— 建议显式契约
`quantile(sorted_latencies, p)` 内部不排序、不校验输入有序，依赖调用方传入升序序列。纯函数契约成立，但建议在 docstring 首行显式声明"输入须为升序"，降低误用风险。

---

## 不变式逐条核对表
| 不变式 | 来源 | 结果 |
|---|---|---|
| A1 限流默认开可关 | 04 §1.2/不变式1 | ✅ 满足（备注见 B-2） |
| A2 p 分位排序取位 | 04 §2.2/不变式2 | ✅ 满足（最近秩法，`quantile` 纯函数） |
| A3 估算带标签 | 04 §3.2/不变式3 | ✅ 满足（estimated=True + "**估算**" 文案） |
| A4 仅成功样本入延迟 | 04 §2.1/不变式4 | ✅ 满足（`lat if ok else None`，统计仅取 success） |
| A5 超时记失败 | 04 §2.1/不变式5 | ⚠️ 功能满足；错误标签冲突（需主管决策） |
| A6 全程只读 GET | 04 §1.3/不变式6 | ✅ 满足（无写方法、无 data=、仅本地落盘） |
| B-A 探测解析解耦纯函数 | 08 §4-A | ✅ 满足（probe/parse_*/quantile 均为纯函数） |
| B-B mock 不经 if 特判 | 08 §4-B | 🔧 已修复（见 [已修复] B-1） |
| B-C header 只从参数 | 08 §4-C | ✅ 满足（无 `os.environ`/硬编码密钥） |
| B-D method 白名单仅 GET/HEAD | 08 §4-D | ✅ 满足（`assert_readonly_method` + 构造层 GET） |

---

## 编译确认
`python -m py_compile src/gateway_monitor.py` → 通过（COMPILE_OK）。
`usage --mock --json`、`bench --mock --json` 全链路离线跑通，退出码 0。
