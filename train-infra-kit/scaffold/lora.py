#!/usr/bin/env python3
"""LoRA 挂载 / 计数 / 保存 / 合并。

依赖（全部在函数内延迟 import，顶层只用标准库）：torch / transformers / peft
安装： pip install -r requirements.txt
声明：本文件在无 GPU、未安装 peft 的沙箱内只做过 py_compile 静态校验。

设计约束（C-A5）：LoRA 的 r/alpha/dropout/target_modules/bias **全部来自 cfg["lora"]**，
本文件不出现任何 target_modules 字面量列表 —— 换挂载层只改 config.yaml。
"""

from typing import Tuple


def build_lora_config(cfg):
    """由 config 组装 peft.LoraConfig；task_type 固定 SEQ_2_SEQ_LM（Whisper 是编码器-解码器）。"""
    from peft import LoraConfig
    lora = cfg["lora"]
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias=str(lora["bias"]),
        task_type="SEQ_2_SEQ_LM",
        inference_mode=False,
    )


def attach_lora(model, cfg):
    """把 LoRA adapter 挂到已加载的 Whisper 上，返回 PeftModel。"""
    from peft import get_peft_model
    peft_model = get_peft_model(model, build_lora_config(cfg))
    peft_model.print_trainable_parameters()      # 打印 trainable/total 与占比，便于核对 r
    return peft_model


def count_trainable(model) -> Tuple[int, int]:
    """返回 (可训练参数量, 总参数量)；占比 = trainable / total，README 引用此口径。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_adapter(model, out_dir):
    """只存 adapter（几 MB），用于后续 merge 或直接加载推理。"""
    model.save_pretrained(str(out_dir))
    return str(out_dir)


def merge_and_save(model, out_dir):
    """合并 adapter 到基座并落盘完整权重。

    注意：merge 后权重精度继承基座。若基座是 fp16 且训练中数值范围偏大，
    合并可能溢出成 inf/nan —— 建议先 merge 到 fp32 再转换，或直接用 save_adapter 保留 adapter。
    """
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    return str(out_dir)
