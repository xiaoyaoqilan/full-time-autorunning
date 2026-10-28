# 知识库结构与填充指南

```
knowledge_base/
├── voice_samples/     ← 【最重要】你的呼吸感原文样本，越多越好
│   ├── README_怎么放.md
│   └── (把 13 篇爆款原文、个人飞轮的 style/voice 文件全部拷进来)
├── articles/          成品长文库（pipeline --gen 的产出也落这里）
│   └── (已灌入：10 篇技术长文 + 五道坎扩张版 + 顶会路线，共 13 篇)
├── topics/
│   └── topic_pool.md  话题池（--gen 取题处，feedback 自动追加衍生题）
└── archive/           高流量回流区（feedback 自动把爆款归档到这里）
```

## voice_samples 怎么放（关键）

1. 把**你本人写的原文**（13 篇爆款、日常推文、公众号稿）原样放进来，`.md` 或 `.txt`，一篇一个文件。
2. 旧机器 personal-flywheel 里的 style/voice 文件如果拿得到，整个目录拷进这里。
3. 放好后跑 `python pipeline.py --refresh-voice`：我会重读全部样本，更新 `config/voice_rules.md` 的规则细节（当前版本是从对话里提炼的，样本级 voice 必须以你原文为准）。

> 红线重申：任何生成/改写前必须实际读取 voice_samples 与 articles 原文，禁止凭记忆即兴模拟。

## 与 WorkBuddy 资料库的关系

本知识库是**本地文件版**（自动化脚本直接读写，这是全自动的前提）。
如果还想要一个云端的、人可以随手翻的镜像（手机上看、跨设备），可以之后把 archive/ 的爆款定期同步进 WorkBuddy 资料库——那是给人看的层，不参与脚本运行。
