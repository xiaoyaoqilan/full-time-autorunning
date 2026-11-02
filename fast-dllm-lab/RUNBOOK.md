# fast-dllm-lab：Diffusion LLM 推理实验手册（5090 单卡）

> 上游：[NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)（Apache-2.0，v1/v2 被 ICLR 2026 收录，NVIDIA Labs）
> 本仓库 vendored 其 v1（`../fast-dllm/v1/`，支持 Dream-7B 与 LLaDA-8B 两族 diffusion LM）。
> 选这条线的理由：**diffusion LLM 是 2026 年最热的新范式之一（ICLR 2026 专题），但做"推理工程"的人极少**——比 nano-vllm 冷门得多，且 NVIDIA Labs 背书过初筛；5090 单卡（32GB）装下 LLaDA-8B/Dream-7B（bf16 14-16GB）绰绰有余。

## 一、背景 60 秒（面试开场用）

自回归（AR）模型一个 token 一个 token 往外蹦；diffusion LM（LLaDA / Dream）把整段回答初始化为 [MASK]，多步迭代去噪、可并行揭示多个 token——理论上天然并行，但实践慢，因为双向注意力没有 KV cache、并行揭示多 token 会破坏依赖。
Fast-dLLM 的两个贡献正好对症：**块级近似 KV cache**（块内复用注意力激活）+ **置信度感知并行解码**（只揭示高置信 token）。官方结果：KV cache 单独 2-3.6×，加并行解码 4-6×，合起来 GSM8K 上最高 11×、全场景最高 27.6× 吞吐（相对 vanilla LLaDA）。

## 二、5090 运行手册

```bash
# 0. 环境
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r fast-dllm/v1/requirements.txt

# 1. 权重（modelscope 国内快）
modelscope download --model AI-ModelScope/LLaDA-8B-Instruct --local_dir ~/huggingface/LLaDA-8B-Instruct
# Dream: modelscope download --model Dream-org/Dream-v0-Instruct-7B --local_dir ~/huggingface/Dream-7B

# 2. 交互冒烟
python fast-dllm/v1/llada/chat.py --gen_length 128 --steps 128 --block_size 32

# 3. 关键参数（实验变量就在这四个里）
#   --gen_length  生成长度        --steps  去噪步数（越小越快越损质量）
#   --block_size  块大小（KV 复用粒度）  --threshold  置信度阈值（并行揭示的门）
#   --use_cache / --if_cache_position  开 KV cache / DualCache

# 4. 精度+吞吐评测（GSM8K / HumanEval 自带脚本）
bash fast-dllm/v1/llada/eval_gsm8k.sh
bash fast-dllm/v1/llada/eval_humaneval.sh
```

## 三、基准回填表（跑完填，简历只引这张表）

| 配置（LLaDA-8B-Instruct, gen 512） | 精度 | 吞吐 (tok/s) | 相对 vanilla 加速 |
|---|---|---|---|
| vanilla（官方，H100） | - | - | 1× |
| +KV cache（官方） | - | - | 2-3.6× |
| +KV+并行解码（官方） | - | - | 最高 11× |
| vanilla（5090 实测） | - | - | - |
| +KV cache（5090） | - | - | - |
| +KV+并行（5090） | - | - | - |
| Qwen3-8B vLLM 同卡对照（质量-速度帕累托锚点） | - | - | - |

## 四、实验计划（每条都是简历素材）

| # | 实验 | 内容 | 为什么有价值 |
|---|---|---|---|
| E1 | 复现 | vanilla/cache/parallel/combined 四配置 × GSM8K/MBPP，精度+吞吐双指标 | 官方只在 H100 发过数，**5090 上没人发过** |
| E2 | 敏感度扫参 | threshold ∈ [0, 0.9]、block_size ∈ {16,32,64}、steps 下探 | 找质量-速度帕累托前沿，产出调参指南 |
| E3 | 对照 AR | 同卡 Qwen3-8B + vLLM 同题对照 | 回答"diffusion LM 推理到底追上 AR 没有"——社区缺这个数 |
| E4 | 消融 | DualCache（缓存 MASK 后缀）开关、置信度阈值与长度关系 | 理解机制而非调参 |
| E5 | 增量改造 | 块间流水（上一块 decode 与下一块 prefill 重叠）/ FP8 双向注意力 | 论文没做的工程点，可发博客/提 PR |

## 五、诚实红线

- 27.6× 是"对比 vanilla LLaDA"的吞吐，**不是对比 vLLM/AR**——diffusion LM 质量仍弱于同级 AR 模型，叙事定位是"新范式的推理工程"，不是"取代 AR"。
- 官方数字来自 H100/A100，5090（Blackwell 消费卡）上没人测过——这是我们的空位，也是风险（可能有 kernel 兼容坑，遇坑即记录）。
- 简历只引本表实测行，回填前不写任何数字。
