# content_flywheel · 全平台内容飞轮

> **当前使用边界（优先于下方历史说明）**：Buffer 推送及 `--publish` 已停用；所有发布由人手动操作。下文关于自动推送、审核通过即推送及 Buffer token 的表述已过时。当前没有 `.env` 或 `novel/bible.md`，因此实际文章与章节生成尚不可用。规则改写和 LLM 打分不能证明技术事实正确，科研内容仍需核查原始来源。`platforms.yaml` 目前不是实际路由程序的唯一事实来源，路由实现见 `adapters.py`。

## 路由矩阵（双 X 账号版）

| 内容类型 | 平台 | 说明 |
|----------|------|------|
| 技术文章 | 知乎 / 小红书 / 领英 / 公众号 / **X 技术号** / 微信读书 | 教授级深度 + 各平台语气改写 |
| 区块链 | **X 区块链号** / 币安广场 | 独立账号，观察视角，不涉个人交易 |
| 高流量 | → archive/ + 话题池衍生选题 | 飞轮回流，自动执行 |

**X 双账号**：`x_tech`（技术号 @ai_xiaomu）与 `x_crypto`（区块链号）完全隔离——类型识别自动路由，技术文绝不进区块链号，反之亦然。`--push --account tech|crypto` 分别用 `BUFFER_PROFILE_ID_TECH / BUFFER_PROFILE_ID_CRYPTO` 两个 profile 推送。

## 新增渠道/模块（2026-10-28 下午）

- **微信读书**：`outbox/weread/`（长文，社区感导语+共鸣结尾，红线同公众号）；`pipeline.py --weread-ideas 3` 每天产 N 条想法到 `outbox/weread_ideas/`（有 LLM 用 LLM+voice，无 LLM 自动降级规则抽取，不断粮）。
- **网文连载引擎 `novel.py`**：`--init "题材"` 生成 bible（人设/世界观/30章大纲/文风契约/伏笔账本）→ `--gen` 日更一章（读 bible+上一章结尾续写，章末必留钩子）→ `--status` 看进度。产出在 `novel/chapters/`。
- **每日一键**：`run_daily.bat` = 网文1章 + 微信读书想法3条 + 技术长文1篇（话题池有空题时）。只生成不发布。
- **定时全自动**：双击 `setup_task.bat` 注册计划任务 ContentFlywheelDaily，每天 08:30 自动跑。卸载：`schtasks /Delete /TN ContentFlywheelDaily /F`。
- **voice_samples 已灌 7 篇呼吸感原文**（3 呼吸版长文 + 4 批次爆款候选库）。配好 .env 后跑 `--refresh-voice` 用样本重提炼规则。



```
话题池(topic_pool.md)
   │ --gen(调LLM+voice规则+样本) 
   ▼
articles/ 长文库
   │ --adapt / --adapt-all(规则改写: 线程/小红书/领英/红线检查)
   ▼
outbox/{zhihu,xhs,x,linkedin,wechat,binance}/   ← 每平台成稿
   │ 发布(X: --push 走Buffer全自动; 其余: 粘贴, 各平台无开放API)
   ▼
线上数据 → feedback/metrics.csv 登记
   │ --feedback(reads≥阈值)
   ▼
archive/ 爆款归档 + 话题池自动追加衍生选题   ← 飞轮闭环
```

## 全自动驾驶（推荐终态：人工只审核）

```
python pipeline.py --autopilot    # 补题→生成→质检门→全平台适配→出一份待审文件
                                   # (人工唯一动作: 打开 review/*_待审.md, 把 status: 待审 改成 ok 或 no)
python pipeline.py --publish      # ok 的自动推 X 对应账号; no 的驳回并清除平台稿
```

- **话题自动补**：待做选题 <3 条时，主编 LLM 按你的人设自动提新题（tech/crypto 各半，自动查重）
- **质检门**：每篇生成后 LLM 打分（准确性/深度/呼吸感/红线），低于 `QUALITY_MIN`（默认 7 分）自动驳回，根本不进审核——人工只看合格品
- **定时化**：双击 `setup_autopilot.bat` 注册 Windows 计划任务（每天 09:00/21:00 生成 + 22:00 回流），之后每天的全部操作 = 打开待审文件改一个词
- 想连审核都免（不建议）：`.env` 里 `AUTO_PUBLISH_X=true` 后，X 推送不再二次确认

## 每日运行（手动版，备用）

```
python pipeline.py --list-topics     # 看今天有什么题
python pipeline.py --gen             # 生成一篇长文入库
python pipeline.py --adapt 2026-10-28_xxx.md   # 一键分发成 5-6 份平台稿
# 打开 outbox 各平台文件夹, 粘贴发布(X 用 --push)
python pipeline.py --feedback        # 有数据后跑回流
```

## 全自动还差什么（三样东西 + 一个确认）

1. **呼吸感原文样本**：把 13 篇爆款原文、个人飞轮的 style/voice 文件放进 `knowledge_base/voice_samples/`，然后跑 `--refresh-voice`。没有样本级 voice，生成只能是"规则级模仿"。
2. **LLM 网关**：`.env` 里填 `LLM_BASE_URL / LLM_API_KEY`（你的 DeepSeek 网关或任一 OpenAI 兼容接口）——`--gen` 的发动机。
3. **Buffer 新 token**：旧 token 已泄露作废，重置后填 `.env`，X 才能真全自动。
4. **确认 X 的路由**：技术+区块链都发，还是只发区块链？

## 自动化边界（诚实版）

- **已全自动**：取题、生成长文、六平台成稿改写、红线拦截、X 推送、流量回流归档、衍生选题。
- **需人工 1 分钟/平台**：知乎/小红书/领英/公众号/币安广场的最终"粘贴发布"——这些平台没有开放发文 API（公众号 API 需认证号配置，知乎可迁回旧机器 zhihu-bot 方案）。
- **防护**：`AUTO_PUSH_X=false` 默认关（遵守"说发才发"）；`gen_daily_limit` 防生成失控；公众号/币安红线词自动拦截。
