# 币安 Agent ID 免费 Mint 清单（2026-10-27 核验）

来源：@congge918 https://x.com/congge918/status/2114920556651073909
链：BNB Smart Chain (56)｜门槛：钱包需持有 .ai Agent ID，开发者模式已开

## 核验结果（22 个）

### ✅ 可铸（免费档确认开放）
| # | 项目 | 免费额度 | 合约 | 备注 |
|---|------|---------|------|------|
| 1 | Black Hole | 前 2 张/钱包 | 0xC84E8281153387e9f35c99cE316075E3c38Dd388 | 已铸 44.6%，进度较快，尽快 |
| 5 | Xuanwu Seed | 第 1 枚免费 | 0x05fd19702029c473683d6188cD856e59038278F4 | 已铸 51.2%，ERC1155，只铸 1 枚（value=0） |
| 6 | Brushborn | 1 张/钱包 | 0x5d43566e40E2e6D396EC10224363E5cAcFc40Cf0 | 水墨风，agent 直接 mint |
| 8 | AGENT 代币 | 9000 枚/Agent ID | 0xA2643B889eEb232708a644f15f70f951773dDEF9 | 公平发射，状态 Open |
| 12 | Pixel Shiba Club | 前 4 次/钱包 | 0xFdD6C5fF6dAbFF40cb27B39D72E5C9517Af27Fdc | 11-1 结束，一次 mint(4) 全拿 |
| 13 | Noir Brokers | 首张免费 | 0xAfE2Becdf86826d4b4341de80a0eb2E98Eec5B02 | 已验证，calldata mint(1) |
| 14 | Floppy Agents | 1 张/钱包 | 0x8F1E4787f131990d793Af253d118aed7Fe1B29AF | calldata 0x1249c58b，value 0 |
| 16 | AISNACK | claim 100 枚 | 0x4903Ce0f68f763B8EdBB80A6E9E836373d3190C8 | ⚠️ Phase1 截止 10-28 00:00 UTC（北京时间 28 日早 8 点），最急 |
| 21 | Agent Periodic Lab | 第 1 包免费 | 0xfE5Ab5ce9b1D86a55d70E7D0573a14D559029C18 | 只领免费 starter pack，别点付费 pack |
| 22 | BeHand Pass | 免费 soulbound 1 张 | 0xCd4Af5a9eDE5ef6f5138028AdC9C152a1Acb80ed | 不可转让，前 2000 张算 Early |

### ⏸ 未开放 / 免费池已空（跳过）
| # | 项目 | 原因 |
|---|------|------|
| 2 | LeadLoop | 官网明确 "Minting is not open yet" |
| 11 | OTV | 500 个免费名额已全部领完（0/500），后续 0.0012 BNB 付费 |
| 19 | PixelVulpes | 官网明确 "Mint not open yet"，0/3333 |

### ❓ 无官网 / 状态未知（让 Agent 预览时自行判断，失败就跳过）
| # | 项目 | 合约 |
|---|------|------|
| 3 | UMBRA Agents | 0x9C0309dC71F89d7f4F0b55d934213C28F10C9e23 |
| 4 | AGENT ID card | 0x486d0947cAa3d70c19e38f6C789489d39291E9e7 |
| 7 | Sonic Protocol | 0xEe581c7D13cfF5c4E34a662ba521C0d3C63EFb6d |
| 9 | Bound Agents | 0xa75526d63bd5E3b3e8A65a5f2efEF45072904d04 |
| 10 | Botmolt | 0x4A15F38c75E529205a45BD2eE57c02AFAB2F6569 |
| 15 | AgentPad Pass | 0xf66fbea12f23D9928D9032bb258Fbc25203957A0 |
| 17 | Lodepass | 0x31b074C8e77e44F29df285F19d4fc76E6D927d01（官网数据加载不出，谨慎） |
| 18 | danta 蛋塔 | 0xcEebc883f922b861841c3e5E47aEb55D3C6f13e9（前 2000 枚免费 FCFS，可能已满） |
| 20 | Alibaba & 70 Thieves | 0xA2F2ffCA66D324af5ED58e5c98c9D6f544b13303 |

## 一键执行提示词（粘贴到币安 App 的 Agent 对话）

```
用币安 Agentic Wallet（baw）在 BSC 按下方清单逐个免费铸造/领取，按顺序执行。

规则（每条都遵守）：
1. 仅打严格免费档（mint/claim 的 value 必须为 0，gas 除外）；不要付费加铸。
2. 每个合约：先 baw contract-call preview，确认无扣款、无多余授权、无恶意风险后，再 execute。
3. 成功把完整合约地址 + txHash 发我；失败/额度满/要付费/报错则跳过该条并说明原因，继续下一条。
4. 需已持有 .ai Agent ID，Developer Mode 已开；未登录先停下来让我扫码登录。
5. 不要对清单外的合约操作；不要 swap/授权无关 spender。

清单（合约地址 → 动作）：
1. 0x4903Ce0f68f763B8EdBB80A6E9E836373d3190C8 → AISNACK claim（免费 100 枚，Phase1 截止 10-28 08:00 北京时间，最优先）
2. 0xC84E8281153387e9f35c99cE316075E3c38Dd388 → Black Hole mint(2)，value 0
3. 0xFdD6C5fF6dAbFF40cb27B39D72E5C9517Af27Fdc → Pixel Shiba freeMint 4 张（一次 mint(4)，value 0）
4. 0x05fd19702029c473683d6188cD856e59038278F4 → Xuanwu Seed 只铸第 1 枚，value 0
5. 0x5d43566e40E2e6D396EC10224363E5cAcFc40Cf0 → Brushborn mint(1)，value 0
6. 0xAfE2Becdf86826d4b4341de80a0eb2E98Eec5B02 → Noir Brokers mint(1)，value 0
7. 0x8F1E4787f131990d793Af253d118aed7Fe1B29AF → Floppy Agents mint()，calldata 0x1249c58b，value 0
8. 0xA2643B889eEb232708a644f15f70f951773dDEF9 → AGENT 代币 claim 9000 枚，value 0
9. 0xfE5Ab5ce9b1D86a55d70E7D0573a14D559029C18 → Agent Periodic Lab 免费领 starter pack（第 1 次铸造），value 0
10. 0xCd4Af5a9eDE5ef6f5138028AdC9C152a1Acb80ed → BeHand Pass claim（soulbound，1 张），value 0

未知状态，逐个尝试，失败跳过：
11. 0x9C0309dC71F89d7f4F0b55d934213C28F10C9e23 → UMBRA Agents
12. 0x486d0947cAa3d70c19e38f6C789489d39291E9e7 → AGENT ID card
13. 0xEe581c7D13cfF5c4E34a662ba521C0d3C63EFb6d → Sonic Protocol
14. 0xa75526d63bd5E3b3e8A65a5f2efEF45072904d04 → Bound Agents
15. 0x4A15F38c75E529205a45BD2eE57c02AFAB2F6569 → Botmolt
16. 0xf66fbea12f23D9928D9032bb258Fbc25203957A0 → AgentPad Pass
17. 0xcEebc883f922b861841c3e5E47aEb55D3C6f13e9 → danta 蛋塔（仅当免费池未满）
18. 0xA2F2ffCA66D324af5ED58e5c98c9D6f544b13303 → Alibaba & 70 Thieves
19. 0x31b074C8e77e44F29df285F19d4fc76E6D927d01 → Lodepass（数据不明，预览异常直接跳过）

确认跳过（勿操作）：LeadLoop 和 PixelVulpes 未开放；OTV 免费池已空。
```

## 风险提示
- 全部为监控扫出来的新项目，免费档只损失 gas（BSC 上一笔约几分钱），但**任何要求 approve 授权的交易直接拒绝**。
- 已核验的 10 个项目官网均标注：value=0、无 owner 提权或仅 pause、源码已验证——预览时再让 Agent 复核一遍。
