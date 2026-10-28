# 05 CodeReview · gpu_snapshot / vram_est / scaffold

> **TL;DR**
> 1. **结论：放行**。5 个 py 全部 `py_compile` 通过；主管裁决清单 7 项逐条核验全部落实。
> 2. 04/09 数值对账一致：GRAD_BPP fp16=2B（裁决 C1）、手算 2.642/5.966/max_batch=111、mock 2 卡零告警退 0、退出码 0/1/2/3 优先级 3>2>1>0、`--util-warn` 默认 90、SSH 零 shell=True、系数单一出处常量表。
> 3. **已修复 3 处**：vram_est 退出码 2 裸字面量改常量；train.py `torch.load` 补 `weights_only=False`（torch≥2.6 续训必挂）；train.py `learning_rate=0` 静默通过补校验。
> 4. **需主管决策 1 项**：fp16 训练路径 `model.half() + GradScaler` 组合（见 §3-D1）。
> 5. 建议不阻塞 6 项（远程模式不采进程表、ARCH_PRESETS 的 custom 键位置等，见 §4）。

---

## 1. 主管裁决清单逐项核验（7/7 通过）

| # | 裁决项 | 代码落点 | 核验结果 |
|---|---|---|---|
| 1 | GRAD_BPP fp16=2B（C1） | `vram_est.py:29` `GRAD_BPP = {"fp32":4, "fp16":2, "bf16":2, "int8":2, "int4":2}` | ✅ 与 04·§2.2 全表一致；ACT_BPP 同表；fp32 时 optimizer master 取 0 防双计（`optimizer_bpp`） |
| 2 | max_batch 装不下输出 0+提示 | `vram_est.py:158-174` `max_batch_size`：`target<=F→(0,NOTE_NO_FIT)`；回退后 `batch<1→(0,NOTE_NO_FIT)`；assumptions 追加该 note | ✅ 实测 `--target-vram 0.9` → `max_batch=0` + 「最小可行配置仍超限…」双出口（渲染层+JSON assumptions） |
| 3 | mock 2 卡 | `gpu_snapshot.py:48-55` `MOCK_SMI_TEXT` 2 行（4090 高载 21600/24564 + 3090 空闲 512/24576） | ✅ 走与线上同一 `parse_smi_csv`；实测默认阈值 `alerts==[]` 退 0；严苛阈值恰 3 条 `(0,temp)(0,util)(0,vram)` 排序与 09·§7 断言一致 |
| 4 | 退出码 0/1/2/3，优先级 3>2>1>0 | `gpu_snapshot.py:27` 常量表；`_Parser.error` 把 argparse exit(2) 统一抛 `UsageError`→3；`main` 唯一出口 | ✅ 实测 `--mock --host x`→3（互斥）、`--nvidia-smi /nonexistent`→2（显式错路径立即失败不回落）、`--host "-oProxyCommand=…"`→3（白名单）、`--host "a b"`→3 |
| 5 | util 告警 `--util-warn` | `DEFAULTS["util_warn"]=90.0`，argparse 默认值引用常量，判定 `>=` 触发、0=关闭 | ✅ 帮助文本、判定、渲染、JSON 回显同源（I4 通过：90/85 字面量仅存在于 DEFAULTS 定义行） |
| 6 | SSH 禁 shell=True、host 白名单 | 全程 argv list；`SMI_REMOTE_CMD` 模块常量恰出现 2 次、零插值；`validate_target` 拒空/拒 `-` 开头/`[A-Za-z0-9._-]{1,253}` fullmatch；`--` 终结符；stderr 剥 ANSI+控制字符截 20 行；`safe_host` 二层防穿越 | ✅ grep 无 `shell=True`/`os.system`/`os.popen`/`getoutput`；`args.mock` 全文件仅 1 处（I2）；纯函数内核无 print/open/subprocess/datetime.now（I1） |
| 7 | 系数单一出处常量表 | `vram_est.py:26-69` PRECISION_BPP/GRAD_BPP/OPTIMIZER_SPEC/CTX/FRAG/RATIO 全表 + 每条 `# 来源：…未实测…±xx%`；`estimate_*` 函数体无裸数字（0/1/GIB 除外）；GIB 唯一换算 | ✅ 04·§7 六条不变式全过；breakdown 求和自检 `gap>=1e-6` 有守卫（`vram_est.py:333-335`） |

## 2. [已修复] 3 处

| # | 文件:行 | 问题 | 修复 |
|---|---|---|---|
| F1 | `vram_est.py:20,356` | 内部错误退出码是裸字面量 `return 2`，与 gpu_snapshot 的 EXIT_* 常量表风格不一致，违反「退出码单一出处」精神 | 新增 `EXIT_INTERNAL = 2` 并引用；docstring 语义不变 |
| F2 | `train.py:300` | `torch.load(..., map_location="cpu")` 未传 `weights_only`：torch≥2.6 默认 `weights_only=True`，而存档含 numpy RNG state（tuple+ndarray）与各类非张量对象，**续训必抛 UnpicklingError**——requirements 只写 `torch>=2.1` 不封顶，属前向兼容逻辑 bug | 改为 `weights_only=False` 并注释理由（存档为本工具自产文件，非不可信来源） |
| F3 | `train.py:157`（validate_config） | RANGES 对 `learning_rate` 下界判 `value < 0`，`lr: 0` 静默通过 → 训练全程零学习率且无任何提示（与 config.yaml 注释「LoRA 常用 1e-4~3e-3」矛盾） | 补显式校验 `learning_rate <= 0` → ConfigError（文案含三条建议）；weight_decay/max_grad_norm 的 0 语义（不衰减/不裁剪）不受影响 |

## 3. [需主管决策] 1 项

| # | 位置 | 问题 | 选项 |
|---|---|---|---|
| D1 | `train.py:248-249, 411-418` | fp16 路径是 `model.half()`（权重/梯度全程 fp16）+ `GradScaler`。GradScaler 的 inf 检查与 `unscale_` 语义按官方文档是配合 **autocast**（权重保持 fp32、计算 fp16）设计的；纯 half 模型 + scaler 可跑，但 fp16 权重下梯度下溢风险更高，且与 `vram_est` 口径「fp16 训练梯度 2B」一致、与「AMP + fp32 master 4B」口径（04·§9-C1 备注）分叉 | (a) 维持现状（纯 half，省显存口径与 vram_est 一致，标注「未实测」）；(b) 改 autocast（权重 fp32，更稳但 W 按精度的估算口径要改）。建议 (a) + README 加「fp16 路径未实测」声明，待上卡后按 04·§8 校准三步法定夺 |

## 4. [建议不阻塞] 6 项

1. **远程模式不采进程表**（`collect_remote` 只跑 `SMI_REMOTE_CMD`）：09·§1.3 未要求 remote 进程列，但 02 若按「三模式同构」断言 processes 会空。属文档间口径问题，建议在 02 补一句。
2. **ARCH_PRESETS 无 `custom` 键**：04·§6 表内含 `"custom": None`，代码改为 choices 里追加 `"custom"`、行为等价（显式给参数即走 custom）。建议回改 04 表注释或代码对齐，二选一。
3. **`train_loop` epoch 计数**：`total_steps` 在 epoch 中途达成时 `epoch += 1` 仍执行，checkpoint 的 epoch 偏大 1；只影响日志/存档元信息，不影响训练正确性。
4. **梯度累积尾批丢弃**：loader 耗尽时不足 `accum` 的残余梯度不 step 直接丢（micro 计数不跨 epoch 重置会累积跨 epoch 补齐，语义上等效但首个 epoch 末可能少一次 step）。脚手架可接受，建议 README「已知未覆盖」补一条。
5. **`_vram_hint` 的 `--target-vram 24` 写死**：OOM 提示命令假设 24G 卡；建议从环境探测或留 TODO。
6. **`set_seed` 返回值无人消费**（`run()` 调用后丢弃返回的快照；存档走 `set_seed_snapshot()`）：无害冗余，可留可删。

## 5. scaffold 结构完整性核验（通过）

- **config 字段 ↔ train.py 引用**：SPECS 六段 34 键与 config.yaml 逐一对齐；`_check_section` 未知键/缺失键/类型/枚举/范围五重校验；`--output-dir/--seed/--resume` 覆盖后仍走全量校验（setdefault 不绕过缺失键检查）。
- **断点续训**：`resolve_resume` auto 取最大 step、path 校验存在性、none 短路；`load_checkpoint` 逐键校验缺失键 + `config_hash`（sha1）不一致拒绝续训；`auto` 找不到档有 warn 不静默。
- **rng 保存**：存档 `rng{python,numpy,torch,cuda}` 四路快照（`set_seed_snapshot`），`restore_rng` 逐键对称恢复；键位清单与 README 契约一致。
- **退出码**：0 正常 / 2 ConfigError·FileNotFoundError / 130 中断，与 epilog 声明一致；`--help` 在无 torch/yaml 裸环境可跑（顶层仅标准库 import）。

## 6. 编译与回归凭证

```
python -m py_compile src/gpu_snapshot.py src/vram_est.py scaffold/train.py scaffold/data.py scaffold/lora.py  → ALL_COMPILE_OK
python src/gpu_snapshot.py --mock                      → exit 0（零告警）
--mock --vram-warn 80 --temp-warn 70 --util-warn 80    → [(0,'temp'),(0,'util'),(0,'vram')]
vram_est --params 0.244 --target-vram 0.9              → max_batch=0 + 「最小可行配置仍超限」
```
