#!/usr/bin/env python3
"""数据管线：manifest -> 音频截断 -> WhisperProcessor 特征化 -> collator。

依赖（全部在函数内延迟 import，顶层只用标准库）：
    torch / transformers / datasets / soundfile / numpy
安装： pip install -r requirements.txt
声明：本文件在无 GPU、未安装上述依赖的沙箱内只做过 py_compile 静态校验。

manifest 格式（JSONL，每行一条）：
    {"audio": "data/wav/0001.wav", "text": "今天天气不错"}
路径相对**当前工作目录**解析，用绝对路径最稳。
"""

import json
from pathlib import Path


def load_manifest(path, audio_column="audio", text_column="text"):
    """读 JSONL -> [{"audio": ..., "text": ...}]；行号级报错，方便定位脏数据。"""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("manifest 不存在：%s" % path)
    rows = []
    for lineno, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("%s 第 %d 行不是合法 JSON：%s" % (path, lineno, exc))
        if audio_column not in item or text_column not in item:
            raise ValueError("%s 第 %d 行缺少 %r 或 %r 键（实际键：%s）"
                             % (path, lineno, audio_column, text_column,
                                ",".join(sorted(item))))
        rows.append({"audio": str(item[audio_column]), "text": str(item[text_column])})
    return rows


def truncate_audio(path, max_seconds, sample_rate):
    """读单声道 float32 波形，超长截断（Whisper 位置编码上限 30s），必要时线性重采样。"""
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:               # 多声道取均值，避免维度不一致
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        # 线性插值重采样：脚手架刻意不引入 resampy/librosa，音质要求高请换 librosa.load
        count = int(round(len(audio) * float(sample_rate) / float(sr)))
        audio = np.interp(np.linspace(0, len(audio) - 1, count),
                          np.arange(len(audio)), audio).astype("float32")
    limit = int(max_seconds * sample_rate)
    return audio[:limit] if len(audio) > limit else audio


def prepare_features(ex, processor, max_seconds=30.0, sample_rate=16000):
    """单条样本 -> {"input_features": [80, 3000], "labels": [T]}（都是 1D，交给 collator 成批）。"""
    audio = truncate_audio(ex["audio"], max_seconds, sample_rate)
    features = processor.feature_extractor(audio, sampling_rate=sample_rate,
                                           return_tensors="pt").input_features[0]
    labels = processor.tokenizer(ex["text"], return_tensors="pt").input_ids[0]
    return {"input_features": features, "labels": labels}


class DataCollatorSpeechSeq2SeqWithPadding:
    """成批：input_features 用 feature_extractor.pad，labels 用 tokenizer.pad 且 pad 位置填 -100。

    - `-100` 是 CrossEntropyLoss 的 ignore_index，保证 padding 不参与 loss（最常见的静默 bug）；
    - labels 首 token 若全是 decoder_start_token，统一切掉，避免重复喂 start（错位一格的经典坑）；
    - `pad_to_multiple_of` 交给 processor 默认配置，保证序列长度对齐张量核心友好。
    """

    def __init__(self, processor, decoder_start_token_id=None):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100)
        start = self.decoder_start_token_id
        if start is not None and labels.shape[1] > 1:
            if bool((labels[:, 0] == start).all().cpu().item()):
                labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_datasets(cfg, processor):
    """构建 train/eval Dataset（已特征化）与 collator；eval_manifest 为 null 时返回 None。"""
    from datasets import Dataset
    data_cfg = cfg["data"]
    max_seconds = float(data_cfg["max_audio_seconds"])
    sample_rate = int(data_cfg["sample_rate"])
    start_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor, start_id)

    def _build(manifest):
        rows = load_manifest(manifest, data_cfg["audio_column"], data_cfg["text_column"])
        dataset = Dataset.from_list(rows)
        return dataset.map(
            lambda ex: prepare_features(ex, processor, max_seconds, sample_rate),
            remove_columns=dataset.column_names,
            desc="feature-extract %s" % Path(manifest).name)

    train_ds = _build(data_cfg["train_manifest"])
    eval_ds = _build(data_cfg["eval_manifest"]) if data_cfg["eval_manifest"] else None
    return train_ds, eval_ds, collator
