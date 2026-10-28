# 05-CodeReview（Review 专员产出）

> **TL;DR**
> - 结论：**放行**。4/4 不变式落实，安全红线通过，Review 专员直接修了 6 处。
> - 已修：F1 存在性 error 被笼统文案掩盖（逻辑 bug）、F2 重复分支、F3 死代码、F4 未用变量、F5 冗余判断、F6 docstring 不符。
> - 安全：仅只读 RPC，无私钥/签名/交易；API key 只从环境变量读，不落盘不打印。
> - 待决策 D1：字节码里函数名是 hex 编码 ASCII，lower() 无法匹配驼峰标识符，属设计文档缺陷。
> - 建议不阻塞 S1（`_POOL` 单例不利单测）、S2（`ff`/`f4` 弱信号几乎必然命中）。
> - **谁该读**：接手改这个代码库前必读，避免踩已知坑；按 F1~F6 定位历史改动原因。


- 审查对象：`src/check_mint.py`（dev 提交版）
- 对照文档：`docs/04-算法设计.md`「dev 必守不变式」4 条
- 审查日期：2026-10-28
- 结论：**放行**（逻辑 bug 已由 Review 专员直接修复，遗留 3 条不阻塞问题）

---

## 一、不变式逐条核对（04-算法设计 §6）

| # | 不变式 | 结论 | 依据 |
|---|---|---|---|
| 1 | max_workers=4 / timeout=10s / 退避 0.5→1→2 / 单合约 60s 预算，默认值不改 | ✅ 落实 | `DEFAULT_WORKERS=4`、`DEFAULT_RPC_TIMEOUT=10.0`、`BACKOFF_BASE=0.5`（factor=2）、`DEFAULT_CONTRACT_BUDGET=60.0`；argparse 仅暴露 `--workers`（范围 1~8，默认 4）。退避计数按逻辑调用计（`rpc_call_with_failover` 换节点也消耗重试次数），1+3=4 次请求上限正确，jitter `uniform(0.8, 1.2)` 符合 ±20% |
| 2 | JSON-RPC error 不换节点不重试；仅网络类错误走 failover + 退避 | ✅ 落实 | `rpc_call_with_failover` 收到含 `error` 字段的正常响应先 `mark_success` 再直接返回 `("rpc_error", …)`；仅 `except Exception`（超时/URLError/429/5xx，`json_rpc_call` 已将 429/5xx 转为异常）走 `mark_failure` + 退避重试 |
| 3 | getCode=="0x" 文案为「EOA 或已自毁合约」，不得断言纯 EOA | ✅ 落实 | `check_existence` 文案「该地址无链上代码（EOA，或合约已自毁）」；评级引擎 reason 亦为「EOA 或已自毁合约」二合一表述 |
| 4 | 所有启发式结论带固定文案；无法确认免费 ⇒ 禁止 ✅ | ✅ 落实 | `CheckResult.heuristic=True` 的条目在报告中统一渲染 HEURISTIC_NOTICE；`rate_contract` 中 `free_confirmed` 仅在 price==0 探测确认时为真，其余一律 ⚠️ 兜底 |

## 二、专项审查

### 2.1 安全（红线项）
✅ **通过**。全文仅 `eth_call` / `eth_getCode` 只读 RPC 与 BscScan `getsourcecode` 查询；无私钥读取、无签名、无交易构造/广播逻辑。BscScan API Key 仅从环境变量读取且只拼入官方 API URL，未落盘、未打印。

### 2.2 并发（线程安全/死锁/竞态）
✅ 无死锁：`NodePool` 单把锁，锁内无嵌套调用、无 I/O。
✅ 竞态可控：失败计数/冷却时间戳的读改写均在 `threading.Lock` 内；`mark_failure` 用 `max()` 合并并发冷却时间，避免旧值覆盖。
✅ 合约级 4 并发 + 节点级串行轮换，与设计 §1.1 一致，无请求级并发。
备注见 4.3-[建议不阻塞]：模块级 `_POOL` 单例（状态本身封装在 NodePool 实例内，非裸全局变量语义，与设计 §1.2「状态存放在 NodePool 中」精神一致）。

### 2.3 异常兜底
✅ 三层兜底完整：`json_rpc_call`（网络错误转异常）→ `check_one_contract`（单检查项 try/except，AC-7）→ `main`（顶层兜底返回码 1）。输入文件 OSError 单独处理返回码 2。可选 BscScan 检查失败不阻塞主流程。

### 2.4 argparse 行为
✅ 位置参数 input 必填；`-o` 缺省输出 stdout；`--workers` choices=1..8 非法值由 argparse 报错退出；文件不可读 → stderr + 退出码 2。行为符合 docstring 声明。

### 2.5 报告格式
✅ 汇总表按风险排序（risk→warn→not_done→ok），逐合约详情含证据与启发式标注，无效输入单独列表，免责声明含设计 §4.3 强制文案。

---

## 三、[已修复]（Review 专员直接修复，共 6 处）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| F1 | `rate_contract`（原 509 行附近） | **逻辑 bug**：`存在性` 检查返回 `error` 时既不计入 `has_error` 也不写 reason，该合约失败原因被「未能探测确认铸造价格为 0」的笼统文案掩盖，排障困难 | 增加 `elif exist.status == "error"` 分支：计入 `has_error` 并输出 `[存在性] <错误信息>` |
| F2 | `rate_contract`（原 529-532 行） | `if r.red_flag` 与 `else` 两分支代码完全相同，冗余且误导（读者以为红旗与非红旗处理不同） | 合并为单一 `reasons.append(f"[{name}] {r.conclusion}")` |
| F3 | `check_interface`（原 341 行） | 死代码 `+ ("" if evid else "")`，且位于 `if not evid` 分支内恒为空串 | 删除 |
| F4 | `check_limit_heuristics`（原 386 行） | `parts` 变量赋值后从未使用 | 删除 |
| F5 | `json_rpc_call`（原 231 行） | `e.code == 503` 被 `500 <= e.code < 600` 完全覆盖，冗余 | 删除冗余判断 |
| F6 | 模块 docstring（不变式 1 描述） | 声称「参数可经 argparse 覆盖」，实际仅 `--workers` 可覆盖，timeout/budget 为常量，描述与实现不符，易误导后续维护者 | 改为「并发数可经 --workers 覆盖（默认 4 不改），其余为代码常量」 |

## 四、[需主管决策]（设计级问题，未擅自改，共 1 条）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| D1 | `BYTECODE_PATTERNS` 的 ASCII 痕迹匹配（`blacklist_marker`/`pause_marker`/`mint_price_marker`） | 设计 §4.2 说「大小写不敏感子串匹配」，但函数名在字节码中是以 **hex 编码的 ASCII 字节** 存在的（如 `mintPrice` → `6d696e745072696365`）。对 hex 串做 lower() 只统一了 hex 字符大小写，**无法**让 `mintPrice`（含大写 P → `50`）匹配 `mintprice`（`70`）。即：混有大小写的标识符（`isBlacklisted`、`MINT_PRICE`、`mintPrice`）实际**匹配不到**，只有源码里恰好全小写（`pause`、`blacklist`）才命中。这是 04-算法设计文档本身的规则缺陷，不是 dev 实现走样 | 漏报：带黑名单/暂停机制的合约可能不被 strong 标记。建议主管决策：改为同时匹配驼峰/下划线各变体的 hex 形式，或在文档中明确降级此组信号为 weak |

## 五、[建议不阻塞]（共 2 条）

| # | 位置 | 说明 |
|---|---|---|
| S1 | 模块级 `_POOL` 单例 | 设计要求「不使用全局变量」，dev 以模块级单例承载、状态严格封装在 NodePool 实例内并加锁，语义合规；但单测时不易替换/重置节点池，建议后续提供注入点（构造函数传参） |
| S2 | `selfdestruct_trace`("ff") / `delegatecall_trace`("f4") 弱信号 | 真实运行时字节码几乎必然包含 `ff`/`f4` 子串，两条 weak 叠加（≥2 → 合并 1 条 ⚠️）是常态，多数合约会带一条「弱信号叠加」提示。属设计文档已明示的「弱信号、误报率高」范畴，仅提示报告噪音，可不处理 |

## 六、验收结论

- 不变式 4/4 落实；安全红线通过；并发与异常兜底无阻塞问题。
- 6 处已修复（1 处逻辑 bug + 5 处清理），1 条设计级问题待主管决策（D1），2 条建议不阻塞（S1/S2）。
- 修复后已通过 `python -m py_compile` 编译验证。

**结论：放行。**
