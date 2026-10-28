# train-infra-kit PRD

> **TL;DR**（≤8 行）
> 1. 目标：给 LLM 算法工程师一套**轻量训练侧 infra**，三件套 = GPU 快照 CLI + 显存估算器 + Whisper+LoRA 脚手架。
> 2. 用户：有 Whisper+LoRA 微调经验、需要把「经验」变成「可复用工程资产」的算法工程师（简历/面试场景）。
> 3. 环境事实：开发沙箱**无 NVIDIA GPU、无 torch**；真实机器可能本机有卡或走局域网远程主机。
> 4. 核心约束：Python 标准库优先，三个 CLI 全部 `--help` 可跑，无卡环境必须能用 `--mock` 演示。
> 5. 取舍：**不做**多机调度、不做 Web 界面、不做自动超参搜索——只做「看得见卡、算得清显存、跑得通模板」。
> 6. 成功标准：新人 clone 后 5 分钟内能回答「我的卡现在怎么样 / 这个配置能不能塞进去 / 训练脚本长什么样」。

---

## 1. 背景与问题

LLM 算法工程师在做 Whisper 微调时，反复踩三类坑：

| 坑 | 现象 | 代价 |
|---|---|---|
| 卡的状态不可见 | 训练前不知道哪张卡空着、显存被谁占着、温度是否已经降频 | 起训 2 分钟后 OOM / 掉卡 |
| 显存靠猜 | 改 `batch_size`、`lora_r`、`seq_len` 全靠试，试一次等 10 分钟 | 反复重启，GPU 小时被浪费 |
| 脚手架每次重写 | 断点续训、日志、LoRA 挂载每换个项目就重抄一遍，还抄漏 | 复现困难，简历上说不清「工程化」 |

三个交付物一一对应。

**非目标用户**：集群管理员（需要调度）、产品/运营（需要 Web）、炼丹大厂（已有成熟平台）。

---

## 2. 交付物与验收标准

验收原则：**每条都是可执行的核对动作**，不是形容词。命令默认在仓库根目录执行。

### 2.1 交付物 A：`src/gpu_snapshot.py` —— GPU 快照 CLI

**定位**：一条命令拿到「现在这台机器/那台机器的卡长什么样」，输出可贴进群、可进 Git 的 markdown + JSON。

#### 功能需求

| 编号 | 需求 | 说明 |
|---|---|---|
| A-F1 | 本机采集 | 调用 `nvidia-smi`，解析 GPU 型号、显存 used/total、利用率、温度、功耗、进程占用 |
| A-F2 | nvidia-smi 路径可配置 | 优先级：`--nvidia-smi <path>` > 环境变量 `NVIDIA_SMI_PATH` > `PATH` 中查找；找不到给出明确报错与三条修复建议 |
| A-F3 | `--mock` 内置样例 | 不依赖任何 GPU/驱动，输出一份固定的 ~~2~4 卡~~ **2 卡**样例数据（含一张高位显存卡、一张高温卡，用于演示告警）（v1.1 修订，经主管裁决 G-01） |
| A-F4 | `--host` 远程采集 | 通过 `ssh <host> nvidia-smi ...` 采集，支持 `--ssh-user` / `--ssh-port` / `--ssh-key`；远程命令失败时错误信息要包含 ssh 返回码与 stderr 前 N 行 |
| A-F5 | 双格式输出 | 默认 markdown 到 stdout；`--json` 输出结构化 JSON；`--out-dir <dir>` 同时落盘 `gpu_snapshot_<host>_<ts>.md` 与 `.json` |
| A-F6 | 阈值告警 | `--vram-warn 90`（显存占用百分比）、`--temp-warn 80`（摄氏度），超阈值在 markdown 中用醒目标记（如 `⚠️`）并在 JSON 中置 `alerts[]`；进程退出码见 A-A6 |
| A-F7 | 纯标准库 | 仅用 `argparse/subprocess/json/datetime/re/sys/os/pathlib/shutil`，无第三方 import |

#### 验收标准（逐条可核对）

- **A-A1** `python src/gpu_snapshot.py --help` 退出码 0，帮助文本中包含 `--mock`、`--host`、`--json`、`--vram-warn`、`--temp-warn`、`--nvidia-smi` 六项。
- **A-A2** 在无 GPU 的开发沙箱执行 `python src/gpu_snapshot.py --mock`，退出码 0，stdout 为合法 markdown，含表头且至少 2 行 GPU 数据。
- **A-A3** `python src/gpu_snapshot.py --mock --json | python -c "import json,sys;d=json.load(sys.stdin);print(len(d['gpus']))"` 输出数字 ≥ 2，说明 JSON 可被标准库解析且顶层含 `gpus` 数组。
- **A-A4** `python src/gpu_snapshot.py --mock --json` 输出的 JSON 中，每条 GPU 记录至少含 `index / name / vram_total_mb / vram_used_mb / util_percent / temp_c` 六个字段；`--mock` 样例中至少触发 1 条 `alerts`。
- **A-A5** `--mock` 与 `--host` 同时给出时报错退出（退出码 2）并提示二者互斥；`--host` 指向不存在的主机时，退出码非 0 且 stderr 含主机名。
- **A-A6** 退出码约定：`0` 全部正常；`1` 采集成功但存在告警；~~`2` 参数错误；`3` 采集失败~~ **`2` 采集失败（nvidia-smi 缺失 / ssh 失败 / 解析失败）；`3` 参数错误**（v1.1 修订，经主管裁决 G-02）。用 `--mock` + 严苛阈值（如 `--vram-warn 1 --temp-warn 1`）验证退出码为 1。
- **A-A7** `--out-dir /tmp/snap` 执行后，该目录下同时存在 1 个 `.md` 和 1 个 `.json`，文件名含主机名与时间戳，且内容与 stdout 一致。
- **A-A8** `grep -nE "^(import|from) " src/gpu_snapshot.py` 结果中不出现 `torch`、`pynvml`、`requests`、`paramiko` 等第三方包。
- **A-A9** `--nvidia-smi /nonexistent/bin/nvidia-smi` 在非 mock 模式下退出码 2（采集失败，v1.1 修订，经主管裁决 G-02），stderr 含该路径原文与「安装 NVIDIA 驱动 / 设置 NVIDIA_SMI_PATH / 改用 --host」三条建议。
- **A-A10** 远程路径在无可用 SSH 环境下允许「无法端到端验证」，但必须提供 `--mock-host` 调试开关或在 README 给出可粘贴的验证命令；实现中 ssh 命令不得通过 `shell=True` 拼接未转义的用户输入。

---

### 2.2 交付物 B：`src/vram_est.py` —— 训练显存估算器

**定位**：**纯数学、零依赖**。训前 5 秒内回答「这套配置能不能塞进我这张卡」。

#### 功能需求

| 编号 | 需求 | 说明 |
|---|---|---|
| B-F1 | 输入参数 | `--params <B>`（模型参数量，单位 B=billion，支持 0.244 这类小数，对应 Whisper-small 244M）、`--precision {fp32,fp16,bf16,int8,int4}`、`--lora-rank <r>`、`--batch-size <n>`、`--seq-len <L>`、`--target-vram <GB>` |
| B-F2 | 分项显存模型 | 分别输出：权重、梯度、优化器状态（AdamW 默认 fp32 master + m + v）、激活、以及固定开销（CUDA context + 碎片）；每项给出公式与数值 |
| B-F3 | LoRA 语义 | 当 `--lora-rank > 0` 时，**不训练全量权重**：梯度/优化器状态只按 LoRA 可训练参数（默认以 `--lora-target-ratio` 近似，默认 0.5%）计算；权重仍按基础精度常驻。`--lora-rank 0` 表示全量微调 |
| B-F4 | 精度换算 | fp32=4B/参数、fp16/bf16=2B、int8=1B、int4=0.5B；量化权重与梯度/优化器的精度关系需在输出中明示假设 |
| B-F5 | 激活估算 | `激活 ≈ k * batch_size * seq_len * hidden * layers * bytes`，`k` 为可调系数（`--act-factor`，默认 4，并说明这是经验值，误差 ±50%） |
| B-F6 | 结论判定 | 与 `--target-vram` 比较，输出 `PASS / TIGHT / OOM` 三态（TIGHT = 估算值落在目标卡的 85%~100%）、显存余量 GB、以及「建议 batch_size 上限」反向推算 |
| B-F7 | 双格式输出 | 默认人类可读表格（含公式列）；`--json` 输出机器可读结构 |
| B-F8 | 纯标准库 | 只用 `argparse/json/sys/dataclasses(可选)/textwrap` |

#### 验收标准

- **B-A1** `python src/vram_est.py --help` 退出码 0，帮助中含全部 6 个必需参数与 `--json`。
- **B-A2** 执行 `python src/vram_est.py --params 0.244 --precision fp16 --lora-rank 8 --batch-size 8 --seq-len 3000 --target-vram 24`，退出码 0，stdout 同时包含「权重」「梯度」「优化器」「激活」四个分项与最终 `PASS/TIGHT/OOM` 之一。
- **B-A3** 单调性：固定其他参数，`--batch-size 1` 的估算总量 < `--batch-size 16` 的总量；`--precision int4` 的总量 < `--precision fp32` 的总量。两条均可用 `--json` 提取 `total_gb` 做数值比较验证。
- **B-A4** `--target-vram 2`（刻意过小）时判定为 `OOM`，且**退出码为 1**；`--target-vram 80` 时判定为 `PASS`，退出码 0。判定三态与退出码一一对应可核对。
- **B-A5** `--lora-rank 0` 与 `--lora-rank 8` 在相同其余参数下，前者总量显著更大（全量梯度+优化器），差异可在 JSON 中核对。
- **B-A6** `--json` 输出含 `breakdown{weights,grads,optimizer,activations,overhead}`、`total_gb`、`verdict`、`headroom_gb`、`max_batch_size`、`assumptions[]` 字段。
- **B-A7** `assumptions[]` 中至少写明三条：激活系数为经验值、优化器按 AdamW、LoRA 可训练参数占比的取法。
- **B-A8** 非法输入（`--params -1`、`--precision fp8`、`--batch-size 0`）退出码 3（参数错误，v1.1 修订，经主管裁决 G-02）且给出具体参数名。
- **B-A9** 源码无第三方 import；`python -c "import ast;ast.parse(open('src/vram_est.py').read())"` 通过。
- **B-A10** 估算误差需声明：文档中注明「与真实 `torch.cuda.max_memory_allocated` 对比，预期误差 ±30%，仅用于预筛，不作为承诺」。

---

### 2.3 交付物 C：`scaffold/` —— Whisper + LoRA 训练脚手架

**定位**：**静态模板交付**，不要求在本沙箱可运行训练。价值在于结构正确、注释到位、换个数据集就能改。

#### 结构要求

```
scaffold/
├── config.yaml            # 单一配置入口：模型/数据/LoRA/训练/日志/断点
├── train.py               # 主入口：argparse + config 加载 + 训练循环
├── data.py                # 数据加载：语音→特征→collator
├── lora.py                # LoRA 挂载/保存/合并
├── requirements.txt       # torch/peft/transformers/datasets 等声明
└── README.md              # 跑起来的三步 + 已知限制
```

#### 功能需求

| 编号 | 需求 | 说明 |
|---|---|---|
| C-F1 | `config.yaml` 单一入口 | 分 `model / data / lora / train / logging / resume` 六段，每段带注释说明取值范围 |
| C-F2 | LoRA 配置 | `r / alpha / dropout / target_modules`（Whisper 的 `q_proj,k_proj,v_proj,out_proj`），并注释「为何 Whisper 常挂注意力投影层」 |
| C-F3 | 断点续训 | `train.py` 支持 `--resume [path|auto]`；保存 `optimizer / scheduler / global_step / epoch / rng` 五要素；`auto` 语义 = 取 `output_dir` 下最大 step |
| C-F4 | 日志 | 同时支持 stdout + 文件（`train.log`）+ 可选 `wandb`（默认关闭）；每 N 步打印 lr/loss/grad_norm/显存峰值 |
| C-F5 | 数据加载 | 基于 `datasets` 的音频字段→`WhisperProcessor` 特征化→`DataCollatorSpeechSeq2SeqWithPadding` 风格 collator；支持 `max_audio_seconds` 截断 |
| C-F6 | 显存相关 | 提供 `gradient_checkpointing`、`gradient_accumulation_steps`、`bf16/fp16` 开关，且注释指向 `vram_est.py` 做预筛 |
| C-F7 | 依赖声明 | `train.py` 顶部注释块写明 `pip install -r requirements.txt`；文件内 import 允许 torch/peft/transformers |

#### 验收标准

- **C-A1** `scaffold/` 下 6 个文件齐备（`config.yaml`、`train.py`、`data.py`、`lora.py`、`requirements.txt`、`README.md`）。
- **C-A2** `config.yaml` 可被标准库解析：`python -c "import yaml"` 不可用时，改用人工核对 6 个顶层段名齐全；若环境有 PyYAML 则 `yaml.safe_load` 通过。
- **C-A3** `python scaffold/train.py --help` 在**无 torch 环境**下仍能打印帮助（import 延迟到 `main()` 内部），退出码 0，含 `--config / --resume / --output-dir`。
- **C-A4** `train.py` 中可 grep 到 `optimizer`、`scheduler`、`global_step`、`epoch`、`rng` 五个断点续训要素的保存与加载逻辑。
- **C-A5** `lora.py` 中可 grep 到 `LoraConfig` 与 `target_modules`，且 `r/alpha/dropout` 从 config 读取而非硬编码。
- **C-A6** `requirements.txt` 含 `torch`、`peft`、`transformers`、`datasets`、`accelerate`，版本为 `>=` 下界而非锁死。
- **C-A7** `README.md` 含「三步跑起来」（装依赖 / 改 config 的 data 段 / `python train.py --config config.yaml`）、以及「本沙箱无 GPU 未验证训练」的显式声明。
- **C-A8** 静态语法自检：`python -m py_compile scaffold/train.py scaffold/data.py scaffold/lora.py` 通过（不导入第三方）。
- **C-A9** `config.yaml` 的 `train` 段含 `gradient_checkpointing`、`gradient_accumulation_steps`、`precision` 三项。
- **C-A10** 全仓 `grep -rn "TODO" scaffold/` 的结果每条都带具体动作（不能是裸 TODO）。

---

## 3. 明确不做（Out of Scope）

| 不做 | 原因 |
|---|---|
| 多机 / 多卡调度（torchrun 编排、Slurm、K8s） | 单机/单机远程已覆盖 90% 场景；调度是独立工程，做了就是半个平台 |
| Web 界面 / 可视化大盘 | CLI + markdown 足够贴群、进 Git；Web 引入前后端与部署成本 |
| 自动超参搜索（Optuna / Ray Tune / 网格搜索） | 与「轻量」冲突；显存估算器已解决「能不能跑」这个前置问题 |
| Prometheus 指标导出 / 告警推送（钉钉、飞书） | 快照 CLI 是拉取式，推送属于监控系统职责 |
| 模型效果评测（WER/CER 流水线） | 属于训练后评估，另一个交付物的范围 |

**灰色地带处理**：`--host` 只做**单主机** ssh 采集，传入多主机列表直接报错并指向「不做多机」章节。

---

## 4. 全局约束

| 约束 | 要求 | 核对方式 |
|---|---|---|
| G1 标准库优先 | `gpu_snapshot.py`、`vram_est.py` **零第三方依赖**；`scaffold/` 例外，允许 torch/peft 等，但必须在文件头注释声明 | A-A8 / B-A9 / C-A6 |
| G2 CLI 自解释 | 三个入口（两个 py + scaffold `train.py`）均支持 `--help`，且 `--help` 不触发重依赖 import | A-A1 / B-A1 / C-A3 |
| G3 无卡可演示 | 无 GPU 环境必须能完成「快照 + 估算」全流程演示（`--mock`） | A-A2 / A-A4 |
| G4 错误可读 | 所有失败路径给出「发生了什么 + 三条怎么办」，禁止裸 traceback 作为唯一输出 | A-A9 / B-A8 |
| G5 输出可复用 | markdown 可直接贴文档，JSON 字段稳定（顶层键不随意改名） | A-A3 / B-A6 |
| G6 Python 版本 | 3.9+，不使用 3.10+ 独有的 `match` / `X \| Y` 类型语法 | `python -m py_compile` |
| G7 无网络依赖 | 除 ssh 采集外，运行期不发起任何网络请求 | 代码审查 |

---

## 5. 交付节奏与验收流程

1. **技术总监** 依据本 PRD 拆分模块与接口（`src/` 两 CLI、`scaffold/` 六文件）。
2. **研发** 实现 → **算法** 校准显存模型的系数与 LoRA 占比假设 → **Review** 过代码。
3. **审核 QA** 逐条执行 §2 的 A/B/C 验收标准，任一失败即打回，验收结论落 `docs/` 下测试记录。
4. **通过线**：A（10 条）+ B（10 条）+ C（10 条）共 30 条，**必须 30/30**；其中 A-A10（远程端到端）允许标记「环境受限，附验证命令」作为有条件通过。

---

## 6. 开放问题（需在开发前确认）

| # | 问题 | 默认决策 |
|---|---|---|
| Q1 | 显存估算的激活系数默认值取多少？ | 默认 4，暴露 `--act-factor`，README 说明校准方法 |
| Q2 | LoRA 可训练参数占比默认 0.5% 是否合理？ | 是（Whisper-small + r=8 量级），暴露 `--lora-target-ratio` |
| Q3 | `--host` 是否需要支持跳板机？ | 不支持，需要时用 `~/.ssh/config` 的 `ProxyJump` |
| Q4 | scaffold 是否要附一份 `train.sh`？ | 不附，避免 shell 与 config 双份真值 |
