# 流水线编排器契约（Orchestrator Contract）

> 用途：把 `company-pipeline` + `prompt-gate` 两份技能里散落的约定，抽成机器可读的单一事实源。
> 本地编排器（pipeline-orchestrator）读这份契约决定「谁先跑、谁是门禁、失败怎么修、状态怎么存、何时人工接管」。
> 来源：`~/.workbuddy/skills/company-pipeline/SKILL.md`、`~/.workbuddy/skills/prompt-gate/SKILL.md`

---

## 1. 角色目录（单一事实源）

每个角色：固定代号、固定产出路径、依赖、是否门禁、是否按需求可选。

| id | 代号 | 产出 | 依赖 | 门禁 | 可选条件 |
|----|------|------|------|------|---------|
| gate | prompt-gate | `docs/00-任务书.md` | 无 | 输入门禁 | 查询类可跳过（主管判断） |
| pm | 产品经理 | `docs/01-PRD.md` | gate | — | 必启 |
| cto | 技术总监 | `docs/02-架构评审.md` | pm | **架构门禁** | 必启 |
| splitter | 任务拆分 | `docs/03-任务拆分.md` | cto | — | 必启 |
| algo | 算法工程师 | `docs/04-算法设计.md` | splitter | — | 有核心逻辑时 |
| infra-serve | 推理 Infra | `docs/08-推理方案.md` | splitter | — | GPU/模型类 |
| infra-gpu | 显卡管控 | `docs/09-GPU管控.md` | splitter | — | 训练/GPU 类 |
| dev | 研发工程师 | `src/` 源码 | algo?, infra-*, splitter | — | 必启 |
| reviewer | Review 专员 | `docs/05-CodeReview.md` | dev | **代码门禁** | 必启 |
| qa | 审核 QA | `docs/06-验收报告.md` | reviewer | **验收门禁** | 必启 |
| media | 媒体专员 | `docs/07-媒体文案.md` | qa(dev并行) | — | 需对外物料时 |
| archivist | 档案员 | `docs/00-摘要索引.md` | qa, media | — | 收尾必启 |

约束：`docs/00-*` 索引永远编号最前，所有角色开工前先读索引与 `00-任务书.md` 定位输入。

## 2. 执行顺序与并行

- 主链：`gate → pm → cto → splitter → dev → reviewer → qa → (media ‖ archivist)`
- 设计岗并行：`algo` 与 `infra-serve`/`infra-gpu` 在 splitter 之后、dev 之前并行启动（dev 等其完成或只读需要的那份）
- 收尾：`media`（如启用）与 `qa` 并行；`archivist` 等 qa（+media）完成

## 3. 质量门禁

| 门禁 | 把关角色 | 通过条件 | 不通过动作 |
|------|---------|---------|-----------|
| 输入门禁 | gate | 产出 00-任务书.md（含验收标准） | 不进流水线 |
| 架构门禁 | 主管读 cto | PRD 可测、范围收敛 | 打回 pm |
| 代码门禁 | reviewer | 无 P0/P1、不变式落实 | 打回 dev，**≤2 轮** |
| 验收门禁 | qa | 验收标准逐条通过 | 打回 dev，**≤2 轮** |

## 4. 失败统一修复机制（主管）

- 任一角色产出判为失败（门禁不通过 / 文档缺失 / 运行异常）→ 标记 `failed` 并落证据路径；
- 打回上限 2 轮：每次打回重新派该角色（dev 或上游）；
- 超过 2 轮 → 触发**主管热修**：主管亲自定位修复，在文档写明「第 N 轮修复 + 原因」，再派 QA 复验一轮终审；
- 热修后仍失败 → 置 `manual_hold`，停止并上报用户。

## 5. 状态管理（落盘，支持中断恢复）

`pipeline.state.json`（项目根目录）结构：

```json
{
  "project": "<slug>",
  "slug": "train-infra-kit",
  "started_at": "2026-10-28T13:00:00",
  "updated_at": "2026-10-28T14:05:00",
  "phase": "running | manual_hold | done",
  "roles": {
    "gate":  {"status": "done",    "attempts": 1, "output": "docs/00-任务书.md", "evidence": ""},
    "pm":    {"status": "done",    "attempts": 1, "output": "docs/01-PRD.md",     "evidence": ""},
    "dev":   {"status": "failed",  "attempts": 3, "output": "src/",                "evidence": "docs/06-验收报告.md#第2轮", "hotfixed": true},
    "qa":    {"status": "running", "attempts": 1, "output": "",                    "evidence": ""}
  },
  "decisions": ["GRAD_BPP fp16=2B", "退出码 采集失败=2/参数错=3"],
  "manual_holds": [],
  "log": ["13:00 gate done", "13:05 pm done", "..."]
}
```

状态枚举：`pending | running | done | failed | manual_hold`。
恢复逻辑：启动时读 state → 已完成角色跳过；`failed/running` 角色重派（保留 attempts 计数）→ 命中打回上限转热修 → 热修失败转 `manual_hold`。

## 6. 人工接管点（保留，不自动执行）

- 外部发布：推文/小红书/公众号发布、Buffer 注入——仅当用户明确说"发"才执行，编排器在 media 之后置 `manual_hold=publish` 等待；
- 账户操作：钱包签名、交易所、SSH 远程写操作——一律 `manual_hold`，由用户亲自做；
- 任何 `manual_hold` 都会停止编排并在日志写明原因，不静默跳过。

## 7. 文档与 token 约定（编排器强制检查）

- 每份产出文档开头 ≤8 行 TL;DR；
- 子 Agent 回报 ≤5 行，禁止贴大段文件；
- 长文/网页先压缩成要点（≤30 行）+ 原文路径再进上下文；
- 收尾必须有 `docs/00-摘要索引.md`，否则 archivist 判未完成。

---

*本契约由「检查两套技能 + 项目约定」阶段产出；编排器 v1 实现以本文件为准。*
