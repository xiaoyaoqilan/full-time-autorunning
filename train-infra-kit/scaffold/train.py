#!/usr/bin/env python3
"""Whisper + LoRA 训练脚手架（结构参考实现）。

运行依赖（本文件顶层不 import 任何第三方库，全部在函数内延迟 import）：
    torch>=2.1  transformers>=4.37  peft>=0.7  datasets>=2.16
    accelerate>=0.25  pyyaml>=6.0  soundfile>=0.12  numpy>=1.24
安装： pip install -r requirements.txt

用法：
    python train.py --config config.yaml
    python train.py --config config.yaml --resume auto --output-dir out/run2

注意：本脚手架在无 GPU、无 torch 的沙箱内只做过静态校验（py_compile + --help），
未验证实际训练收敛；真实训练前请先用 src/vram_est.py 预筛显存。
"""

import argparse
import json
import os
import sys
from pathlib import Path

SECTIONS = ("model", "data", "lora", "train", "logging", "resume")

# 字段 -> 允许类型；未知键与缺失键都会报错（防拼写错误静默生效）
SPECS = {
    "model": {"name_or_path": (str,), "language": (str, type(None)), "task": (str,),
              "freeze_encoder": (bool,)},
    "data": {"train_manifest": (str,), "eval_manifest": (str, type(None)),
             "audio_column": (str,), "text_column": (str,), "sample_rate": (int,),
             "max_audio_seconds": (int, float), "num_workers": (int,)},
    "lora": {"r": (int,), "alpha": (int,), "dropout": (int, float),
             "target_modules": (list,), "bias": (str,)},
    "train": {"output_dir": (str,), "per_device_batch_size": (int,),
              "gradient_accumulation_steps": (int,), "learning_rate": (int, float),
              "weight_decay": (int, float), "warmup_ratio": (int, float),
              "num_epochs": (int, float), "max_steps": (int,), "precision": (str,),
              "gradient_checkpointing": (bool,), "max_grad_norm": (int, float),
              "seed": (int,), "save_steps": (int,), "save_total_limit": (int,)},
    "logging": {"log_file": (str,), "log_steps": (int,), "wandb": (dict,)},
    "resume": {"mode": (str,), "path": (str, type(None))},
}
WANDB_SPEC = {"enabled": (bool,), "project": (str,), "run_name": (str, type(None))}
PRECISIONS = ("bf16", "fp16", "no")
LORA_BIASES = ("none", "all", "lora_only")
RESUME_MODES = ("none", "auto", "path")

RANGES = (   # (段, 字段, 下界, 上界, 是否闭区间上界)
    ("data", "sample_rate", 1, None, True),
    ("data", "max_audio_seconds", 0, 30, True),
    ("data", "num_workers", 0, 16, True),
    ("lora", "r", 0, 64, True),
    ("lora", "alpha", 1, None, True),
    ("lora", "dropout", 0, 0.3, True),
    ("train", "per_device_batch_size", 1, None, True),
    ("train", "gradient_accumulation_steps", 1, None, True),
    ("train", "learning_rate", 0, None, False),
    ("train", "weight_decay", 0, None, True),
    ("train", "warmup_ratio", 0, 0.2, True),
    ("train", "num_epochs", 0, None, False),
    ("train", "max_grad_norm", 0, None, True),
    ("train", "save_steps", 1, None, True),
    ("train", "save_total_limit", 0, None, True),
    ("logging", "log_steps", 1, None, True),
)


class ConfigError(Exception):
    """配置类错误：文案必须是「发生了什么 + 三条怎么办」。"""


def load_config(path):
    """读 YAML；import yaml 延迟到此处，保证 --help 在无 PyYAML 环境也能跑。"""
    target = Path(path)
    if not target.is_file():
        raise ConfigError("配置文件不存在：%s\n"
                          "  1) 用 --config <path> 指定配置\n"
                          "  2) 复制 scaffold/config.yaml 后按需修改\n"
                          "  3) 在 scaffold/ 目录下直接 python train.py" % path)
    try:
        import yaml
    except ImportError:
        raise ConfigError("缺少 PyYAML\n"
                          "  1) pip install -r requirements.txt\n"
                          "  2) 或 pip install 'pyyaml>=6.0'\n"
                          "  3) 确认当前 python 就是训练用的解释器")
    try:
        cfg = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:                       # YAML 语法错误
        raise ConfigError("YAML 解析失败（%s）：%s\n"
                          "  1) 检查缩进是否为两空格、有无 Tab\n"
                          "  2) 检查冒号后是否有空格\n"
                          "  3) 用 python -c \"import yaml;yaml.safe_load(open('%s'))\" 定位"
                          % (type(exc).__name__, exc, path))
    if not isinstance(cfg, dict):
        raise ConfigError("配置顶层必须是映射（六段：%s）\n"
                          "  1) 确认文件不是空文件\n"
                          "  2) 确认没有把整段包在列表里\n"
                          "  3) 对照 scaffold/config.yaml 模板" % "/".join(SECTIONS))
    return cfg


def _check_section(cfg, section, spec):
    block = cfg.get(section)
    if not isinstance(block, dict):
        raise ConfigError("缺少配置段 %s 或它不是映射\n"
                          "  1) 六段必须齐全：%s\n"
                          "  2) 段名顶格写、子键两空格缩进\n"
                          "  3) 对照 scaffold/config.yaml" % (section, "/".join(SECTIONS)))
    unknown = sorted(set(block) - set(spec))
    if unknown:
        raise ConfigError("%s 段存在未知键：%s\n"
                          "  1) 检查拼写（例如 learning_rate 而非 learnig_rate）\n"
                          "  2) 未知键不会被忽略，删掉或改名\n"
                          "  3) 字段清单见 scaffold/README.md" % (section, ", ".join(unknown)))
    missing = sorted(set(spec) - set(block))
    if missing:
        raise ConfigError("%s 段缺少必填键：%s\n"
                          "  1) 从 scaffold/config.yaml 补齐\n"
                          "  2) 值为 null 也要显式写出\n"
                          "  3) 不要注释掉整行" % (section, ", ".join(missing)))
    for key, types in spec.items():
        value = block[key]
        if isinstance(value, bool) and bool not in types:
            raise ConfigError("%s.%s 不接受布尔值，当前 %r" % (section, key, value))
        if not isinstance(value, types):
            raise ConfigError("%s.%s 类型错误，期望 %s，当前 %r\n"
                              "  1) 数字不要加引号\n"
                              "  2) 列表用 '-' 起行的块风格\n"
                              "  3) 见 scaffold/config.yaml 同名注释"
                              % (section, key, "/".join(t.__name__ for t in types), value))
    return block


def validate_config(cfg):
    """纯函数：六段齐全 + 类型 + 范围 + 未知键 + 枚举值。返回 None，失败抛 ConfigError。"""
    if not isinstance(cfg, dict):
        raise ConfigError("配置顶层必须是映射")
    extra = sorted(set(cfg) - set(SECTIONS))
    if extra:
        raise ConfigError("存在未知配置段：%s\n"
                          "  1) 只允许 %s\n"
                          "  2) 段名顶格、无缩进\n"
                          "  3) 见 scaffold/config.yaml" % (", ".join(extra), "/".join(SECTIONS)))
    for section in SECTIONS:
        _check_section(cfg, section, SPECS[section])
    _check_section(cfg["logging"], "wandb", WANDB_SPEC)
    if cfg["logging"]["wandb"]["enabled"] and not cfg["logging"]["wandb"]["project"]:
        raise ConfigError("logging.wandb.enabled 为 true 时必须给 project")
    for section, key, low, high, closed in RANGES:
        value = cfg[section][key]
        if low is not None and value < low:
            raise ConfigError("%s.%s 不能小于 %s（当前 %r）" % (section, key, low, value))
        if high is not None and (value > high if closed else value >= high):
            raise ConfigError("%s.%s 必须%s %s（当前 %r）"
                              % (section, key, "不超过" if closed else "小于", high, value))
    if cfg["train"]["learning_rate"] <= 0:
        raise ConfigError("train.learning_rate 必须大于 0（当前 %r）\n"
                          "  1) LoRA 常用 1e-4~3e-3，全量微调用 1e-5 量级\n"
                          "  2) YAML 里写 0.001 而非 0\n"
                          "  3) 对照 scaffold/config.yaml 同名注释"
                          % cfg["train"]["learning_rate"])
    if cfg["model"]["task"] not in ("transcribe", "translate"):
        raise ConfigError("model.task 只能是 transcribe / translate")
    if cfg["lora"]["bias"] not in LORA_BIASES:
        raise ConfigError("lora.bias 只能是 %s" % " / ".join(LORA_BIASES))
    if cfg["train"]["precision"] not in PRECISIONS:
        raise ConfigError("train.precision 只能是 %s" % " / ".join(PRECISIONS))
    if cfg["resume"]["mode"] not in RESUME_MODES:
        raise ConfigError("resume.mode 只能是 %s" % " / ".join(RESUME_MODES))
    if cfg["resume"]["mode"] == "path" and not cfg["resume"]["path"]:
        raise ConfigError("resume.mode=path 时必须给 resume.path")
    modules = cfg["lora"]["target_modules"]
    if not modules or not all(isinstance(m, str) and m for m in modules):
        raise ConfigError("lora.target_modules 必须是非空的字符串列表\n"
                          "  1) 用块风格列表（每行 '    - q_proj'）\n"
                          "  2) 常见值：q_proj/k_proj/v_proj/out_proj\n"
                          "  3) 空列表等价于不挂 LoRA，请把 lora.r 设为 0")
    if cfg["data"]["sample_rate"] != 16000:
        sys.stderr.write("[warn] data.sample_rate=%s，Whisper 期望 16000，将发生重采样\n"
                         % cfg["data"]["sample_rate"])
    return None


def build_argparser():
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Whisper + LoRA 训练脚手架（config 为唯一真值，命令行只做覆盖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("示例：\n"
                "  python train.py --config config.yaml\n"
                "  python train.py --config config.yaml --resume auto\n"
                "  python train.py --config config.yaml --resume out/run/checkpoint-500.pt\n"
                "退出码：0 正常结束 / 2 配置或数据错误 / 130 用户中断。\n"))
    p.add_argument("--config", default="config.yaml", help="配置文件路径，默认 config.yaml")
    p.add_argument("--resume", default=None, metavar="{none,auto,PATH}",
                   help="none 从头训 / auto 取 output_dir 下最大 step 存档 / 或直接给 .pt 路径")
    p.add_argument("--output-dir", default=None, help="覆盖 train.output_dir")
    p.add_argument("--seed", type=int, default=None, help="覆盖 train.seed")
    return p


def set_seed(seed):
    """设置四路随机源并返回快照（存档用，保证续训可复现）。"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    """按四元组逐键恢复随机源（与 set_seed 对称）。"""
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def config_hash(cfg):
    """配置指纹：续训时若配置已变，拒绝加载（避免形状/超参错配的静默错误）。"""
    import hashlib
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def build_model(cfg):
    """加载 Whisper、可选冻结 encoder、按 config 挂 LoRA。"""
    import torch
    from transformers import WhisperForConditionalGeneration
    from lora import attach_lora
    model_cfg = cfg["model"]
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["name_or_path"])
    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False          # checkpointing 与 KV cache 互斥
    if model_cfg["freeze_encoder"]:
        model.model.encoder.requires_grad_(False)
    if cfg["lora"]["r"] > 0:
        model = attach_lora(model, cfg)
    if cfg["train"]["precision"] == "fp16":
        model = model.half()                    # bf16 交给 autocast，避免手工转精度
    return model


def build_optim_sched(model, cfg, total_steps):
    """AdamW 只吃 requires_grad 参数；线性 warmup + 线性衰减。"""
    import torch
    from transformers import get_linear_schedule_with_warmup
    train_cfg = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ConfigError("没有可训练参数：检查 lora.r 与 model.freeze_encoder 的组合")
    optimizer = torch.optim.AdamW(params, lr=train_cfg["learning_rate"],
                                 weight_decay=train_cfg["weight_decay"])
    warmup = int(total_steps * train_cfg["warmup_ratio"])
    # lr 未按 batch 线性缩放：config 里的 learning_rate 是最终值，缩放请在配置层决定
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, max(1, total_steps))
    return optimizer, scheduler


def save_checkpoint(path, model, optimizer, scheduler, scaler, cfg, step, epoch):
    """存档契约：键位固定，load_checkpoint 逐键读回（见 02·2.6.4）。"""
    import torch
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(step),
        "epoch": float(epoch),
        "rng": set_seed_snapshot(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config_hash": config_hash(cfg),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    return Path(path)


def set_seed_snapshot():
    """当前随机源快照（不重置种子）；与 restore_rng 对称。"""
    import random
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def load_checkpoint(path, model, optimizer, scheduler, scaler, cfg):
    """与 save_checkpoint 对称：逐键 ckpt["x"] 读回，配置指纹不一致则拒绝续训。"""
    import torch
    # weights_only=False：存档含 numpy RNG state（tuple+ndarray），torch>=2.6 默认
    # weights_only=True 会拒绝加载导致续训失败；存档是本工具自己写的文件，非不可信来源。
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "global_step", "epoch", "rng",
                "scaler", "config_hash"):
        if key not in ckpt:
            raise ConfigError("存档缺少键 %s：%s（存档不完整，换一个 step 或从头训）"
                              % (key, path))
    if ckpt["config_hash"] != config_hash(cfg):
        raise ConfigError("配置与存档指纹不一致，拒绝续训：%s\n"
                          "  1) 用训练该存档时的 config 续训\n"
                          "  2) 或改 resume.mode=none 从头训\n"
                          "  3) 或把 output_dir 指到新目录" % path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    restore_rng(ckpt["rng"])
    return int(ckpt["global_step"]), float(ckpt["epoch"])


def resolve_resume(mode, output_dir, path=None):
    """auto = output_dir 下 checkpoint-<step>.pt 的最大 step；找不到返回 None（由调用方告警）。"""
    if mode == "none":
        return None
    if mode == "path":
        target = Path(path or "")
        if not target.is_file():
            raise ConfigError("resume.path 不存在：%s\n"
                              "  1) 检查路径（形如 out/whisper-lora/checkpoint-500.pt）\n"
                              "  2) 存档文件名必须是 checkpoint-<step>.pt\n"
                              "  3) 或改 resume.mode=auto" % path)
        return target
    out = Path(output_dir)
    if not out.is_dir():
        return None
    best, best_step = None, -1
    for item in out.glob("checkpoint-*.pt"):
        try:
            step = int(item.stem.split("-")[-1])
        except ValueError:
            continue
        if step > best_step:
            best, best_step = item, step
    return best


def prune_checkpoints(output_dir, keep):
    """只删比当前小的档，保留最近 keep 个（keep=0 表示不清理）。"""
    if keep <= 0:
        return []
    items = []
    for item in Path(output_dir).glob("checkpoint-*.pt"):
        try:
            items.append((int(item.stem.split("-")[-1]), item))
        except ValueError:
            continue
    items.sort()
    removed = []
    for _, item in items[:-keep]:
        item.unlink()
        removed.append(item.name)
    return removed


class Tee:
    """同时写 stdout 与日志文件（train.log），行首带时间戳。"""

    def __init__(self, path):
        from datetime import datetime
        self._dt = datetime
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._fh.write("\n==== run %s ====\n" % self._dt.now().strftime("%Y-%m-%d %H:%M:%S"))

    def log(self, msg):
        line = "[%s] %s" % (self._dt.now().strftime("%H:%M:%S"), msg)
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def _vram_hint(cfg):
    """OOM 时给出可直接粘贴的 vram_est 命令（C-F6）。"""
    arch = "whisper-small"
    name = str(cfg["model"]["name_or_path"]).lower()
    for key in ("large-v3", "medium", "small", "base", "tiny"):
        if key in name:
            arch = "whisper-" + key
            break
    seq = int(cfg["data"]["max_audio_seconds"] * 50 * 2)      # 30s ≈ 3000 token(enc+dec)
    return ("python src/vram_est.py --arch %s --precision %s --batch-size %d --seq-len %d "
            "--target-vram 24 --json" % (arch, cfg["train"]["precision"],
                                         cfg["train"]["per_device_batch_size"], seq))


def train_loop(cfg, model, loader, optimizer, scheduler, scaler, log, start_step, start_epoch,
               total_steps, device):
    """主循环：梯度累积 -> 裁剪 -> step；每 log_steps 打印指标，每 save_steps 存档。"""
    import torch
    train_cfg = cfg["train"]
    accum = train_cfg["gradient_accumulation_steps"]
    step, epoch, micro = start_step, start_epoch, 0
    model.train()
    model.zero_grad(set_to_none=True)
    while step < total_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=train_cfg["precision"] == "bf16"):
                    loss = model(**batch).loss / accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except torch.cuda.OutOfMemoryError:
                log.log("显存不足（OOM）：先降 train.per_device_batch_size 或 seq/act 相关项")
                log.log("用下面命令重算显存：")
                log.log("  " + _vram_hint(cfg))
                raise
            micro += 1
            if micro % accum:
                continue
            if scaler is not None:
                scaler.unscale_(optimizer)
            if train_cfg["max_grad_norm"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg["max_grad_norm"])
            else:
                grad_norm = torch.tensor(0.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % train_cfg["save_steps"] == 0:
                path = Path(train_cfg["output_dir"]) / ("checkpoint-%d.pt" % step)
                save_checkpoint(path, model, optimizer, scheduler, scaler, cfg, step, epoch)
                removed = prune_checkpoints(train_cfg["output_dir"],
                                            train_cfg["save_total_limit"])
                log.log("存档 %s%s" % (path.name,
                                      ("；清理 " + ",".join(removed)) if removed else ""))
            if step % cfg["logging"]["log_steps"] == 0:
                peak = (torch.cuda.max_memory_allocated() / (2 ** 30)
                        if torch.cuda.is_available() else 0.0)
                log.log("step %d/%d loss %.4f lr %.3e grad_norm %.3f peak %.2f GiB"
                        % (step, total_steps, float(loss) * accum,
                           scheduler.get_last_lr()[0], float(grad_norm), peak))
            if step >= total_steps:
                break
        epoch += 1
        if cfg["data"]["eval_manifest"]:
            log.log("epoch %d 结束（评测请接 evaluate，脚手架未内置指标计算）" % epoch)
    return step, epoch


def run(cfg):
    """编排：seed -> processor -> model -> data -> optim -> resume -> loop。"""
    import torch
    from transformers import WhisperProcessor
    from data import build_datasets
    train_cfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = Tee(Path(train_cfg["output_dir"]) / cfg["logging"]["log_file"])
    log.log("device=%s precision=%s lora.r=%d output=%s"
            % (device, train_cfg["precision"], cfg["lora"]["r"], train_cfg["output_dir"]))
    log.log("config_hash=%s" % config_hash(cfg))
    set_seed(train_cfg["seed"])
    processor = WhisperProcessor.from_pretrained(cfg["model"]["name_or_path"],
                                                 language=cfg["model"]["language"],
                                                 task=cfg["model"]["task"])
    model = build_model(cfg).to(device)
    train_ds, _eval_ds, collator = build_datasets(cfg, processor)
    if len(train_ds) == 0:
        raise ConfigError("训练集为空：检查 data.train_manifest 是否为空文件"
                          "（每行一条 audio/text 的 JSON 对象）")
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_cfg["per_device_batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=collator,
        pin_memory=(device.type == "cuda"))
    steps_per_epoch = max(1, len(loader) // train_cfg["gradient_accumulation_steps"])
    total_steps = (train_cfg["max_steps"] if train_cfg["max_steps"] > 0
                   else int(round(steps_per_epoch * train_cfg["num_epochs"])))
    optimizer, scheduler = build_optim_sched(model, cfg, total_steps)
    scaler = None
    if train_cfg["precision"] == "fp16":
        try:
            scaler = torch.amp.GradScaler("cuda")        # torch>=2.3
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()         # 旧版回退
    ckpt = resolve_resume(cfg["resume"]["mode"], train_cfg["output_dir"],
                          cfg["resume"]["path"])
    start_step, start_epoch = 0, 0.0
    if ckpt is not None:
        start_step, start_epoch = load_checkpoint(ckpt, model, optimizer, scheduler,
                                                  scaler, cfg)
        log.log("续训自 %s（global_step=%d）" % (Path(ckpt).name, start_step))
    elif cfg["resume"]["mode"] == "auto":
        log.log("[warn] resume.mode=auto 但 %s 下没有 checkpoint-*.pt，将从 step 0 开始"
                % train_cfg["output_dir"])
    step, epoch = train_loop(cfg, model, loader, optimizer, scheduler, scaler, log,
                             start_step, start_epoch, total_steps, device)
    final = Path(train_cfg["output_dir"]) / ("checkpoint-%d.pt" % step)
    save_checkpoint(final, model, optimizer, scheduler, scaler, cfg, step, epoch)
    log.log("训练结束：step=%d，最终存档 %s" % (step, final.name))
    log.close()
    return 0


def main(argv=None):
    """顶层不导入第三方库：--help 在无 torch/yaml 环境也必须退出码 0。"""
    args = build_argparser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.output_dir:
            cfg.setdefault("train", {})["output_dir"] = args.output_dir
        if args.seed is not None:
            cfg.setdefault("train", {})["seed"] = args.seed
        if args.resume:
            if args.resume in ("none", "auto"):
                cfg["resume"] = dict(cfg.get("resume") or {}, mode=args.resume)
            else:
                cfg["resume"] = {"mode": "path", "path": args.resume}
        validate_config(cfg)
        return run(cfg)
    except ConfigError as exc:
        sys.stderr.write("配置错误：%s\n" % exc)
        return 2
    except FileNotFoundError as exc:
        sys.stderr.write("文件不存在：%s\n"
                         "  1) 检查 data.train_manifest 路径\n"
                         "  2) 路径相对当前工作目录，不是相对 config 文件\n"
                         "  3) 用绝对路径最稳\n" % exc)
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("已中断：最后一档 checkpoint 仍在 output_dir 下，可用 --resume auto 恢复\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
