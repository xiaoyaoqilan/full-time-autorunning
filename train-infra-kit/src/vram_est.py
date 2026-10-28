#!/usr/bin/env python3
"""训练显存估算（交付物 B）。

    W = P_total x bpp / GIB        G = P_trainable x gbpp / GIB
    O = P_trainable x (master+state) / GIB
    A = act_factor x b x seq x hidden x layers x abpp / GIB
    OH = ctx + frag x (W+G+O)      T = W + G + O + A + OH（单位 GiB = 2^30 B）

LoRA 单真值：P_trainable = P_total x lora_target_ratio（--lora-rank 0 视为全量）；
rank 不进入任何估算函数，只做全量开关与渲染层一致性提示。
退出码：0 PASS / 1 TIGHT 或 OOM / 2 内部错误 / 3 参数错误。
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION, EXIT_OK, EXIT_NOT_FIT, EXIT_INTERNAL, EXIT_USAGE = 1, 0, 1, 2, 3
GIB = float(2 ** 30)     # 唯一换算常量，任何函数不得再写 1024**3 / 2**30
BILLION = float(1e9)     # --params 输入单位 B（十亿）-> 个，仅此一处

# 来源：数据类型位宽（确定值）：fp32 4B / fp16,bf16 2B / int8 1B / int4 0.5B。
# 量化 scale 开销（nf4 双重量化，≈0.004~0.13 B/param）对 0.244B 模型 <0.03 GiB，忽略。
PRECISION_BPP = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}
# 来源：梯度跟随训练计算精度（主管裁决 C1：fp16/bf16 取 2 B，与 02·2.5.4 手算表一致）。
# fp32 分支为 dtype 位宽（确定值）；低精度分支为「梯度与参数同精度」惯例，本仓库未实测，误差 ±20%。
GRAD_BPP = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 2.0, "int4": 2.0}
# 来源：激活与梯度同精度口径（惯例，本仓库未实测，误差 ±20%），fp32 训练为 4 B。
ACT_BPP = GRAD_BPP
# 来源：AdamW 的 m/v 为 fp32 共 8 B（框架惯例，确定值）；非 fp32 另加 4 B fp32 master
# （混合精度惯例）；8bit 变体 int8 m/v = 2 B，分块 absmax scale 忽略，未实测，误差 ±10%。
OPTIMIZER_SPEC = {"adamw": {"state_bpp": 8.0, "master_bpp": 4.0},
                  "adamw_8bit": {"state_bpp": 2.0, "master_bpp": 4.0}}
DEFAULT_OPTIMIZER = "adamw"   # --optimizer 未暴露为 CLI 参数，预留表键即可扩展（04·§2.3）
# 来源：经验值（本仓库未实测）。依据：每 token 每层需驻留 attention 输出、MLP 中间张量与反向所需
# 输入副本，共约 3~5 份；未开 flash-attention 偏保守、开 gradient checkpointing 偏激进，误差 ±50%。
DEFAULT_ACT_FACTOR = 4.0
# 来源：经验值（本仓库未实测）。CUDA context + cuBLAS workspace + 驱动预留，常见 0.3~0.7 GiB，
# 随驱动/CUDA 版本浮动，误差 ±40%（小模型场景主导误差）。
CTX_OVERHEAD_GIB = 0.5
# 来源：经验值（本仓库未实测）。caching allocator 常见 3%~8%；口径按 04·§9-C3 冻结为
# 「基数不含激活」，代价是系统性低估 3%~4%（单向偏低），收益是 OH 与 batch 无关。
FRAG_RATIO = 0.05
# 来源：Q2 决策默认（非实测）。Whisper-small r=8 全挂约 0.73%、只挂 self-attn 约 0.48%，
# 取 0.5% 为保守折中；LoRA 下 G+O 仅占总量约 0.6%，ratio 误差 50% 只影响 T 0.3%。
DEFAULT_LORA_RATIO = 0.005

DEFAULT_HIDDEN, DEFAULT_LAYERS = 768, 12   # Whisper-small 单侧结构默认（R1）
DEFAULT_BATCH, DEFAULT_SEQ_LEN = 1, 1500   # seq = 单次前向 token 总数（encoder+decoder）
MAX_BATCH_CAP = 4096      # 防空卡/巨大 target 下输出无意义大数
PASS_RATIO, TIGHT_RATIO = 0.85, 1.00

# (params[B], hidden, layers)：取自各模型公开配置，参数量为约数
ARCH_PRESETS = {"whisper-tiny": (0.039, 384, 4), "whisper-base": (0.074, 512, 6),
                "whisper-small": (0.244, 768, 12), "whisper-medium": (0.769, 1024, 24),
                "whisper-large-v3": (1.55, 1280, 32)}
# 公式单一出处：渲染层只读此表，防文档与代码漂移（R15）
FORMULAS = {"weights": "W = P_total x bpp(precision) / GIB",
            "grads": "G = P_trainable x gbpp(precision) / GIB",
            "optimizer": "O = P_trainable x (master_bpp + state_bpp) / GIB",
            "activations": "A = act_factor x batch x seq x hidden x layers x abpp / GIB",
            "overhead": "OH = ctx_overhead + frag_ratio x (W + G + O)"}
LABELS = {"weights": "权重", "grads": "梯度", "optimizer": "优化器",
          "activations": "激活", "overhead": "固定开销"}
ORDER = ("weights", "grads", "optimizer", "activations", "overhead")
NOTE_NO_FIT = "最小可行配置仍超限，需降 precision / 缩 seq_len / 换卡"
NOTE_CAPPED = "max_batch_size 已触顶 MAX_BATCH_CAP=%d，实际可再增大" % MAX_BATCH_CAP

class UsageError(Exception):
    """参数错误 -> exit 3。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

@dataclass
class EstInput:
    params: float            # 参数总量（个）
    hidden: int
    layers: int
    precision: str
    lora_rank: int
    lora_ratio: float
    batch_size: int
    seq_len: int
    act_factor: float
    target_vram: float
    optimizer: str = DEFAULT_OPTIMIZER

def normalize(params: float, hidden: int, layers: int, precision: str, lora_rank: int,
              lora_ratio: float, batch_size: int, seq_len: int, act_factor: float,
              target_vram: float, optimizer: str = DEFAULT_OPTIMIZER) -> EstInput:
    """L0 输入校验 + 单位归一；非法输入点名参数（B-A8）。"""
    if precision not in PRECISION_BPP:
        raise UsageError("--precision 只支持 %s，收到 %r"
                         % ("/".join(sorted(PRECISION_BPP)), precision))
    if optimizer not in OPTIMIZER_SPEC:
        raise UsageError("--optimizer 只支持 %s，收到 %r"
                         % ("/".join(sorted(OPTIMIZER_SPEC)), optimizer))
    for name, value in (("--params", params), ("--target-vram", target_vram),
                        ("--hidden", hidden), ("--layers", layers), ("--seq-len", seq_len),
                        ("--batch-size", batch_size), ("--act-factor", act_factor)):
        if value is None or value <= 0:
            raise UsageError("%s 必须为正数（当前 %r）" % (name, value))
    if lora_rank < 0:
        raise UsageError("--lora-rank 不能为负（当前 %r，0 表示全量微调）" % lora_rank)
    if not 0 < lora_ratio <= 1:
        raise UsageError("--lora-target-ratio 必须在 (0,1] 区间（当前 %r）" % lora_ratio)
    return EstInput(float(params) * BILLION, int(hidden), int(layers), precision,
                    int(lora_rank), float(lora_ratio), int(batch_size), int(seq_len),
                    float(act_factor), float(target_vram), optimizer)

def split_params(e: EstInput) -> Tuple[float, float, float]:
    """L1 拆分 -> (P_total, P_trainable, P_frozen)；rank 只做全量开关，ratio 是唯一真值。"""
    ratio = 1.0 if e.lora_rank == 0 else e.lora_ratio
    trainable = e.params * ratio
    return e.params, trainable, e.params - trainable

def optimizer_bpp(e: EstInput) -> float:
    """L2 优化器字节数：fp32 训练时 master 与权重共用，故取 0（避免重复计账）。"""
    spec = OPTIMIZER_SPEC[e.optimizer]
    return (0.0 if e.precision == "fp32" else spec["master_bpp"]) + spec["state_bpp"]

def estimate_weights(e: EstInput) -> float:
    return split_params(e)[0] * PRECISION_BPP[e.precision] / GIB

def estimate_grads(e: EstInput) -> float:
    return split_params(e)[1] * GRAD_BPP[e.precision] / GIB

def estimate_optimizer(e: EstInput) -> float:
    return split_params(e)[1] * optimizer_bpp(e) / GIB

def estimate_activations(e: EstInput) -> float:
    return (e.act_factor * e.batch_size * e.seq_len * e.hidden * e.layers
            * ACT_BPP[e.precision] / GIB)

def estimate_overhead(weights: float, grads: float, optimizer: float) -> float:
    """固定开销只吃 W/G/O，与 batch 无关（保证 T 对 b 线性、闭式可解）。"""
    return CTX_OVERHEAD_GIB + FRAG_RATIO * (weights + grads + optimizer)

def estimate_total(e: EstInput) -> Tuple[float, Dict[str, float]]:
    """汇总 (total_gb, breakdown)；breakdown 顺序即渲染顺序，分项之和必须等于总量。"""
    w, g, o = estimate_weights(e), estimate_grads(e), estimate_optimizer(e)
    breakdown = {"weights": w, "grads": g, "optimizer": o,
                 "activations": estimate_activations(e),
                 "overhead": estimate_overhead(w, g, o)}
    return sum(breakdown.values()), breakdown

def verdict(total_gb: float, target_vram: float) -> str:
    """三态判定：端点归属 ratio==0.85 -> PASS，ratio==1.00 -> TIGHT，>1.00 -> OOM。"""
    ratio = total_gb / target_vram
    if ratio <= PASS_RATIO:
        return "PASS"
    return "TIGHT" if ratio <= TIGHT_RATIO else "OOM"

def max_batch_size(e: EstInput, target_vram: float) -> Tuple[int, Optional[str]]:
    """L4 闭式初值 + 用 L2 的 estimate_total 逐点回退校验（禁止在此重写公式）。"""
    fixed = estimate_total(replace(e, batch_size=0))[0]    # 激活为 0，OH 与 b 无关
    step = estimate_activations(replace(e, batch_size=1))  # 单样本边际成本
    if step <= 0:
        return MAX_BATCH_CAP, NOTE_CAPPED
    if target_vram <= fixed:
        return 0, NOTE_NO_FIT
    batch = min(int((target_vram - fixed) // step), MAX_BATCH_CAP)
    while batch > 0 and estimate_total(replace(e, batch_size=batch))[0] > target_vram:
        batch -= 1                                          # 浮点/取整方向修正
    while (batch < MAX_BATCH_CAP
           and estimate_total(replace(e, batch_size=batch + 1))[0] <= target_vram):
        batch += 1
    if batch < 1:
        return 0, NOTE_NO_FIT
    return batch, (NOTE_CAPPED if batch >= MAX_BATCH_CAP else None)

def _g(value: float) -> str: return "%.6g" % value

def substitutions(e: EstInput) -> Dict[str, str]:
    """四列表中的「代入值」列：与 FORMULAS 一一对应，全部由常量表渲染。"""
    trainable = split_params(e)[1]
    w, g, o = estimate_weights(e), estimate_grads(e), estimate_optimizer(e)
    return {
        "weights": "%se9 x %s B" % (_g(e.params / BILLION), _g(PRECISION_BPP[e.precision])),
        "grads": "%se6 x %s B" % (_g(trainable / 1e6), _g(GRAD_BPP[e.precision])),
        "optimizer": "%se6 x %s B" % (_g(trainable / 1e6), _g(optimizer_bpp(e))),
        "activations": "%s x %d x %d x %d x %d x %s B" % (
            _g(e.act_factor), e.batch_size, e.seq_len, e.hidden, e.layers,
            _g(ACT_BPP[e.precision])),
        "overhead": "%s + %s x (%s + %s + %s)" % (
            _g(CTX_OVERHEAD_GIB), _g(FRAG_RATIO), _g(w), _g(g), _g(o))}

def rank_hint(e: EstInput) -> str:
    """rank 只出现在渲染层：一行一致性提示，不参与计算。"""
    if e.lora_rank == 0:
        return "全量微调（--lora-rank 0）：P_trainable = P_total，ratio 不生效"
    return ("LoRA r=%d 全挂（q/k/v/out + cross-attn）约 0.73%%，只挂 self-attn 约 0.48%%；"
            "本次按 ratio=%.4g 计算，估算偏保守" % (e.lora_rank, e.lora_ratio))

def build_assumptions(e: EstInput, extra: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """结构化 assumptions（R10）：消费方按 .key 判定，不 grep 自由文本。"""
    return [
        {"key": "act_factor", "value": e.act_factor,
         "note": "经验值，本仓库未实测；含义 = 每 token 每层保留的中间张量份数（含反向所需），误差 ±50%"},
        {"key": "optimizer", "value": e.optimizer,
         "note": "AdamW 的 m/v 为 fp32 共 8 B，非 fp32 训练另加 4 B fp32 master；LoRA 下 G+O 仅占总量约 0.6%，误差 ±10%"},
        {"key": "lora_ratio", "value": (1.0 if e.lora_rank == 0 else e.lora_ratio),
         "note": "唯一真值：P_trainable = P_total x ratio；r=8 全挂约 0.73%、只挂 self-attn 约 0.48%，0.5% 为保守折中；rank 不进入公式"},
        {"key": "grad_bpp", "value": GRAD_BPP[e.precision],
         "note": "梯度跟随训练精度（fp32=4 B、低精度=2 B）；fp32 为 dtype 位宽确定值，低精度分支为惯例，本仓库未实测，误差 ±20%"},
        {"key": "overhead", "value": {"ctx_gib": CTX_OVERHEAD_GIB, "frag_ratio": FRAG_RATIO},
         "note": "上下文 0.5 GiB（经验值，未实测，误差 ±40%）+ 碎片 5%（未实测）；碎片基数按 04·§9-C3 不含激活，系统性低估 3%~4%"},
        {"key": "error_budget", "value": "±30%~±35%",
         "note": "激活 ±50% 传导到总量约 ±31%，只可用于预筛；±50% 是激活系数误差、±30% 是总量误差，两者不矛盾"},
    ] + extra

def render_table(e: EstInput, total_gb: float, breakdown: Dict[str, float],
                 target_vram: float, batch: int, batch_note: Optional[str],
                 assumptions: List[Dict[str, object]]) -> str:
    subs = substitutions(e)
    out = ["显存估算（单位 GiB = 2^30 B）", "",
           "| 分项 | 公式 | 代入值 | GB (GiB) |", "|---|---|---|---|"]
    out += ["| %s | %s | %s | %.3f |" % (LABELS[k], FORMULAS[k], subs[k], breakdown[k])
            for k in ORDER]
    out += ["| **合计** | T = W + G + O + A + OH | — | **%.3f** |" % total_gb, "",
            "判定：%s（ratio = %.3f，目标 %.3g GiB，余量 %.3f GiB）"
            % (verdict(total_gb, target_vram), total_gb / target_vram, target_vram,
               target_vram - total_gb),
            "建议 max_batch_size = %d（不预留 85%% 余量；建议实际留 10~15%%）" % batch,
            "结构：params=%s B hidden=%d layers=%d precision=%s batch=%d seq=%d rank=%d"
            % (_g(e.params / BILLION), e.hidden, e.layers, e.precision, e.batch_size,
               e.seq_len, e.lora_rank),
            rank_hint(e)]
    if batch_note:
        out.append("提示：" + batch_note)
    out += ["", "假设（assumptions）："] + ["- [%s] %s" % (a["key"], a["note"])
                                        for a in assumptions]
    return "\n".join(out) + "\n"

def to_json_dict(e: EstInput, total_gb: float, breakdown: Dict[str, float], batch: int,
                 assumptions: List[Dict[str, object]]) -> Dict[str, object]:
    total, trainable, frozen = split_params(e)
    return {"schema_version": SCHEMA_VERSION,
            "input": {"params": round(e.params / BILLION, 6), "params_total": total,
                      "params_trainable": trainable, "params_frozen": frozen,
                      "hidden": e.hidden, "layers": e.layers, "precision": e.precision,
                      "lora_rank": e.lora_rank,
                      "lora_target_ratio": 1.0 if e.lora_rank == 0 else e.lora_ratio,
                      "batch_size": e.batch_size, "seq_len": e.seq_len,
                      "act_factor": e.act_factor, "target_vram_gb": e.target_vram,
                      "optimizer": e.optimizer},
            "breakdown": {k: round(v, 6) for k, v in breakdown.items()},
            "total_gb": round(total_gb, 6), "target_vram_gb": e.target_vram,
            "ratio": round(total_gb / e.target_vram, 6),
            "verdict": verdict(total_gb, e.target_vram),
            "headroom_gb": round(e.target_vram - total_gb, 6),
            "max_batch_size": batch, "assumptions": assumptions}

class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        raise UsageError(message)

def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="vram_est.py",
        description="训练显存估算：权重/梯度/优化器/激活/固定开销五层 + LoRA 修正 + max_batch 反解",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("退出码：0 PASS / 1 TIGHT 或 OOM / 3 参数错误\n"
                "示例：\n"
                "  python src/vram_est.py --params 0.244 --target-vram 24\n"
                "  python src/vram_est.py --arch whisper-medium --batch-size 8 "
                "--seq-len 3000 --target-vram 24 --json\n"
                "单位 GiB = 2^30 B；总量误差 ±30%（激活系数误差 ±50%），仅用于预筛。\n"))
    p.add_argument("--params", type=float, default=None, metavar="B",
                   help="模型总参数量，单位 B（十亿），如 0.244；必填，除非给 --arch")
    p.add_argument("--target-vram", type=float, default=None, metavar="GIB",
                   help="目标卡显存 GiB，如 24；必填")
    p.add_argument("--precision", default="fp16", choices=sorted(PRECISION_BPP),
                   help="训练精度，默认 fp16（决定权重/梯度/激活 bpp）")
    p.add_argument("--lora-rank", type=int, default=8,
                   help="LoRA rank，默认 8；0 = 全量（不参与显存计算，仅做一致性提示）")
    p.add_argument("--lora-target-ratio", type=float, default=DEFAULT_LORA_RATIO,
                   help="可训练参数占比，默认 " + str(DEFAULT_LORA_RATIO) + "（唯一真值）")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="per-device batch size，默认 %d" % DEFAULT_BATCH)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                   help="单次前向参与激活的 token 总数（encoder + decoder）；"
                        "Whisper 30s = 3000 mel 帧 -> 1500 encoder + 1500 decoder ≈ 3000；"
                        "默认 %d" % DEFAULT_SEQ_LEN)
    p.add_argument("--hidden", type=int, default=None,
                   help="hidden 维，默认 %d（Whisper-small）" % DEFAULT_HIDDEN)
    p.add_argument("--layers", type=int, default=None,
                   help="单侧 transformer 层数（enc/dec 同构按单侧计），默认 %d" % DEFAULT_LAYERS)
    p.add_argument("--arch", default=None, choices=sorted(list(ARCH_PRESETS) + ["custom"]),
                   help="结构预设；与显式 --params/--hidden/--layers 并存时显式值优先")
    p.add_argument("--act-factor", type=float, default=DEFAULT_ACT_FACTOR,
                   help="激活系数 k，默认 " + str(DEFAULT_ACT_FACTOR)
                        + "（经验值，本仓库未实测，误差 ±50%%）")
    p.add_argument("--json", action="store_true", help="输出 JSON（默认四列 markdown 表格）")
    return p

def _resolve_arch(args: argparse.Namespace) -> List[Dict[str, object]]:
    """--arch 与显式值并存时显式优先，并在 assumptions 记录覆盖关系（R1）。"""
    extras: List[Dict[str, object]] = []
    preset = ARCH_PRESETS.get(args.arch) if args.arch else None
    if preset:
        overrides = []
        for i, key in enumerate(("params", "hidden", "layers")):
            if getattr(args, key) is None:
                setattr(args, key, preset[i])
            else:
                overrides.append("--" + key)
        extras.append({"key": "arch", "value": args.arch,
                       "note": "预设 %s = (params %s B, hidden %d, layers %d)%s"
                               % (args.arch, _g(preset[0]), preset[1], preset[2],
                                  "；显式覆盖 " + "/".join(overrides) if overrides
                                  else "；无显式覆盖")})
    if args.params is None:
        raise UsageError("--params 必填（或改用 --arch whisper-small 等预设）")
    if args.target_vram is None:
        raise UsageError("--target-vram 必填（无目标卡无法给出有意义结论）")
    args.hidden = DEFAULT_HIDDEN if args.hidden is None else args.hidden
    args.layers = DEFAULT_LAYERS if args.layers is None else args.layers
    return extras

def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        extras = _resolve_arch(args)
        est = normalize(args.params, args.hidden, args.layers, args.precision, args.lora_rank,
                        args.lora_target_ratio, args.batch_size, args.seq_len,
                        args.act_factor, args.target_vram)
        total_gb, breakdown = estimate_total(est)
        gap = abs(sum(breakdown.values()) - total_gb)
        if gap >= 1e-6:                      # 不变量：分项之和必须等于总量
            raise RuntimeError("breakdown 校验失败：分项和与 total_gb 差 %.3g GiB" % gap)
        batch, batch_note = max_batch_size(est, est.target_vram)
        if batch_note:
            extras.append({"key": "max_batch_size", "value": batch, "note": batch_note})
        assumptions = build_assumptions(est, extras)
        payload = (json.dumps(to_json_dict(est, total_gb, breakdown, batch, assumptions),
                              ensure_ascii=False, indent=2) if args.json
                   else render_table(est, total_gb, breakdown, est.target_vram, batch,
                                     batch_note, assumptions))
        sys.stdout.write(payload if payload.endswith("\n") else payload + "\n")
        sys.stdout.flush()
        return EXIT_OK if verdict(total_gb, est.target_vram) == "PASS" else EXIT_NOT_FIT
    except UsageError as exc:
        sys.stderr.write("用法错误：%s\n  1) python src/vram_est.py --help\n"
                         "  2) --params 与 --target-vram 为必填\n"
                         "  3) 数值参数必须为正，精度取值见 --help\n" % exc.message)
        return EXIT_USAGE
    except SystemExit as exc:
        return int(exc.code or EXIT_OK)
    except Exception as exc:                 # 不打印 traceback
        sys.stderr.write("内部错误：%s: %s\n" % (type(exc).__name__, exc))
        return EXIT_INTERNAL

if __name__ == "__main__":
    raise SystemExit(main())
