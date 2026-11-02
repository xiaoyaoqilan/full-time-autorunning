# 架构对照：DeepSeek Super-agent × pipeline-runner

> 2026-11-02 ｜ 输入：DeepSeek《Super-agent 架构》文章（Router/Worker 两层、主动环境感知、任务线连续、后台任务、云脑+本地执行、Watchdog）
> 结论：五点里三点本地编排器已覆盖，两点（Watchdog、通知钩子）本次 v1.1 补齐；一项（跨平台任务线）属平台级能力，明确不做。

## 逐点对照

| DeepSeek 观点 | pipeline-runner 现状 | v1.1 动作 |
|---|---|---|
| 1. Router 收需求、Worker 并行干活 | ✅ 编排器=Router（依赖调度+线程池并行），AI 会话按角色=Worker | 无需改 |
| 2. 没人 @ 也主动查问题（环境感知） | ❌ 需要消息/CI/邮件接入，属平台级 | **不做**（边界：外部发布与账户操作保持人工接管） |
| 3. 任务线跨平台连续（群聊↔GitHub↔邮件） | ❌ 同上，本地无多入口路由 | **不做**（SKILL.md 的落盘续命已保证"换会话不换上下文"） |
| 4. 后台跑任务不挡新活 + 分场合回复 | ✅ 并行调度已覆盖；回复风格由各角色 prompt 约束 | 无需改 |
| 5. 状态与执行分离 + Watchdog 盯进度 | ⚠️ 状态落盘已有；无活性检测 | ✅ **新增**：heartbeat/<stage> 心跳（执行期每 5s 刷新）+ `watch` 看门狗（心跳超时 → 额度内自动重新入队 / 用完转 FAILED 等主管 retry）+ `run` 启动时先恢复僵死残留 |
| （隐含）任务结束主动通知 | ❌ | ✅ **新增**：定义级 `notify_cmd` 模板（{stage}/{status}/{reason}），done/retry/failed/watchdog_* 均触发 |
| 代价：多会话 token、注入风险 | Token 守卫已有（8/12/16 报进度）；notify_cmd 是命令注入面 | **红线**：notify_cmd 只能来自可信的流水线定义文件，禁止把用户消息拼进模板 |

## v1.1 实测记录

- 崩溃恢复：运行中 `kill -9` 编排器 → state 停在 running、心跳停更 → `watch --stall-sec 3` 检出"心跳超时 15s"→ 重试额度用完转 FAILED → `retry` 可恢复。全链路通过。
- 并发保护：另一进程执行中（心跳新鲜）时，`status`/`watch` 不会误重置 running（心跳新鲜度判定，>15s 才算死）。
- 修复的两个 bug：① `_load` 无条件重置 running 会误伤活跃执行 → 改为心跳判定；② 恢复逻辑原在 `_load` 抢跑导致 watchdog 分支永不触发 → 统一收敛到 `_recover_stale(stall_sec)`，run 与 watch 共用。

## 与 DeepSeek 设计的本质差异

DeepSeek 是云端常驻服务（云脑 + 数据库 + 多设备本地代理）；pipeline-runner 是**单机可恢复的状态机**——没有常驻服务，"长期在线"靠落盘续命 + 人工/定时触发 `watch`/`run`。它的 Router/Worker/Watchdog 分层在编排器里是同构的，但消息接入层（群聊/邮件/GitHub 监听）明确留在平台侧。
