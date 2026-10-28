# Whisper + LoRA 训练脚手架

「配置即真值」的最小可运行骨架：`config.yaml` 定义一切，`train.py` 只做读取、校验与编排。
结构完整、注释到位，可直接替换数据路径后开跑。

> **重要声明**：本脚手架在无 GPU、未安装 torch/peft 的沙箱内只做了静态校验
> （`python -m py_compile`、`python train.py --help`、YAML 结构检查），
> **未验证实际训练与收敛**。首次上卡请先用小样本（`max_steps: 20`）跑通全链路。

## 依赖

| 包 | 下界 | 用途 |
|---|---|---|
| `torch` | 2.1 | 训练框架 |
| `transformers` | 4.37 | `WhisperForConditionalGeneration` / `WhisperProcessor` / 线性 warmup |
| `peft` | 0.7 | LoRA 注入与合并 |
| `datasets` | 2.16 | JSONL -> Dataset + map 特征化 |
| `accelerate` | 0.25 | 设备放置（可选，单卡不依赖） |
| `pyyaml` | 6.0 | 读取 `config.yaml`（延迟 import，缺它 `--help` 仍可用） |
| `soundfile` | 0.12 | 读 wav/flac 波形 |
| `numpy` | 1.24 | 重采样与波形运算 |

## 三步跑起来

```bash
# 1) 装依赖（建议独立虚拟环境）
pip install -r requirements.txt

# 2) 改 config.yaml 的 data 段：train_manifest 指向你的 jsonl
#    每行一条：{"audio": "/abs/path/0001.wav", "text": "今天天气不错"}

# 3) 启动
python train.py --config config.yaml
```

常用覆盖（不改配置文件）：

```bash
python train.py --config config.yaml --output-dir out/run2 --seed 7
python train.py --config config.yaml --resume auto                 # 取最大 step 存档续训
python train.py --config config.yaml --resume out/run2/checkpoint-500.pt
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `config.yaml` | 唯一真值；六段 `model/data/lora/train/logging/resume`，每字段带范围注释 |
| `train.py` | 参数解析 -> 配置校验 -> 延迟 import -> 组模型/数据/优化器 -> 训练循环 -> 存档 |
| `data.py` | manifest 读取、音频截断、特征化、`-100` padding 的 collator |
| `lora.py` | `LoraConfig` 组装（全部来自 config）、挂载、可训练参数计数、保存/合并 |
| `requirements.txt` | 依赖下界清单 |

## 关键约定

- **顶层只 import 标准库**：`yaml` 在 `load_config()` 内、`torch/transformers/peft` 在各自函数内导入，
  因此 `python train.py --help` 在裸环境下退出码 0。
- **未知键报错**：`learnig_rate` 这类拼写错误会直接抛「未知键」，不会静默用默认值。
- **存档契约**：`checkpoint-<global_step>.pt`，键位固定
  `model / optimizer / scheduler / global_step / epoch / rng{python,numpy,torch,cuda} / scaler / config_hash`。
  续训时配置指纹（sha1）不一致会拒绝加载，避免形状或超参错配的静默错误。
- **`resume.mode=auto`** 取 `output_dir` 下最大 step 的存档；找不到会打印 warn 并从 step 0 开始（不静默）。
- **LoRA 参数只由 config 决定**：换 `target_modules` 只改 YAML，代码里没有字面量列表。
  只挂 self-attn 约 0.48% 可训练参数，连 cross-attn 全挂约 0.73%。

## 显存与 OOM

先用估算器预筛，再上机（估算总量误差 ±30%，仅用于预筛）：

```bash
python ../src/vram_est.py --arch whisper-small --precision bf16 \
  --batch-size 8 --seq-len 3000 --target-vram 24
```

训练中 OOM 时 `train_loop` 会打印可直接粘贴的 `vram_est` 命令。常见处置顺序：
降 `train.per_device_batch_size` → 开 `gradient_checkpointing` → 降精度 → 缩 `data.max_audio_seconds`。

## 显存与精度（bf16 / fp16）

- **bf16 可用时优先 bf16**（Ampere 及以上，如 A100/H100/4090）：指数位与 fp32 相同，动态范围大、
  不需要 loss scaling，数值上比 fp16 稳得多。`config.yaml` 直接设 `train.precision: bf16` 即可。
- **fp16 仅用于不支持 bf16 的旧卡**（Turing 及更早），此时走标准 AMP 做法：
  前向/损失在 `torch.autocast(device_type="cuda", dtype=torch.float16)` 内，
  反向梯度由 `torch.cuda.amp.GradScaler` 缩放，优化器步进用 `scaler.step(optimizer)` +
  `scaler.update()`（步进前先 `scaler.unscale_(optimizer)` 做梯度裁剪），避免 fp16 梯度下溢。
- bf16 与 fp16 不要同时开：`train.precision` 三选一（`fp32 / bf16 / fp16`）。

## 已知未覆盖

- 未实现评测指标（CER/WER）：`eval_manifest` 只保证数据可构建，指标请自行接 `evaluate` 或 `jiwer`。
- 未接 `wandb`：`logging.wandb.enabled` 目前只是开关位，接入点留在 `run()` 的日志行。
- 重采样是线性插值（刻意不引 `librosa`），对音质敏感的场景请替换 `data.truncate_audio`。
