#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""平台适配器: 一篇技术长文 → 各平台成稿(规则版, 无需 LLM; 需要 LLM 精修时走 pipeline --gen 的文本再润)"""
import re

WECHAT_REDLINES = ["投资建议", "荐股", "翻墙", "VPN", "男女对立", "宗教", "政治敏感"]
CRYPTO_HINTS = ["比特币", "区块链", "Solana", "以太", "crypto", "稳定币", "链上", "DeFi",
                "NFT", "币安", "BTC", "ETH", "SOL", "代币", "挖矿", "钱包"]


def _title(text):
    m = re.search(r"^#\s+(.+)", text, re.M)
    return m.group(1).strip() if m else "未命名"


def _sections(text):
    """[(标题, 正文), ...]"""
    parts = re.split(r"^## ", text, flags=re.M)
    out = []
    for p in parts[1:]:
        lines = p.splitlines()
        out.append((lines[0].strip(), "\n".join(lines[1:]).strip()))
    return out


def to_zhihu(text):
    """知乎: 长文直发, 保留全部深度。只加导语。"""
    return f"{text}\n\n---\n(原创技术长文, 欢迎评论区拍砖)"


def to_wechat(text):
    """公众号: 长文直发, 但硬红线检查。命中红线则拒绝产出并说明。"""
    title = _title(text)
    for w in WECHAT_REDLINES:
        if w in text:
            return f"[红线拦截] 命中公众号红线词: {w}。改写去掉相关表述后再发, 或放弃此平台。"
    return text  # 公众号只发长文/图片帖, 原文即稿


def _clean(line):
    """去 markdown 痕迹: 加粗、标题符、表格行、分隔线"""
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"`(.+?)`", r"\1", line)
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"^[-*]\s+", "· ", line)
    return line.strip()


def _body_lines(text, limit=None):
    """清洗后的正文行(去掉标题/表格/空行)"""
    out = []
    for l in text.splitlines():
        l = _clean(l)
        if not l or l.startswith("|") or l == "---" or l.startswith("· —"):
            continue
        out.append(l)
        if limit and len(out) >= limit:
            break
    return out


def to_x(text):
    """X: 线程版。钩子帖 = 标题+第一句; 正文按章节拆帖, 每帖≤240字。"""
    title = _title(text)
    body = _body_lines(text)
    hook = ""
    for l in body[1:]:
        if l == title or len(l) < 12 or l in ("一句话", "引言", "前言") \
           or re.match(r"^[一二三四五六七八九十\d]+、", l):
            continue
        hook = l[:90] + ("…" if len(l) > 90 else "")
        break
    posts = [f"{title}\n\n{hook} 🧵"]
    for sec_title, sec_body in _sections(text):
        lines = _body_lines(sec_body)
        chunk, cur = [], 0
        for l in lines:
            if cur + len(l) > 220 and chunk:
                break
            chunk.append(l)
            cur += len(l)
        if chunk:
            posts.append(f"{sec_title}\n\n" + "\n".join(chunk[:6]))
    posts.append("完整长文在知乎/公众号。大伙最想深挖哪一段, 评论区说。")
    return "\n\n---\n\n".join(posts)


def to_xhs(text):
    """小红书: 标题悬念化(20字内), 正文共鸣+口语化改写提示, 末尾标签。"""
    t = _title(text)
    hook = f"终于有人把{t[:14]}讲明白了"
    body_lines = []
    for sec_title, body in _sections(text)[:4]:
        lines = _body_lines(body, limit=1)
        if lines:
            body_lines.append(f"✅ {sec_title}\n{lines[0][:120]}…")
    body = (f"{hook}\n\n看这篇就够了。\n\n" + "\n\n".join(body_lines) +
            "\n\n完整推导版放在图里了, 收藏慢慢看。\n\n"
            "#AI #大模型 #程序员 #技术分享 #人工智能")
    return body


def to_linkedin(text):
    """领英: 结论先行, 三条 takeaways, 职业化口吻。"""
    secs = _sections(text)[:3]
    takes = "\n".join(f"{i+1}. {st}" for i, (st, _) in enumerate(secs))
    return (f"关键词: {_title(text)}\n\n"
            f"核心结论:\n{takes}\n\n"
            f"完整机制拆解(公式/账本/代码级)见长文版。欢迎工程师与研究者交流。\n\n"
            f"#AI #LLM #Engineering")


def to_weread(text):
    """微信读书长文: 书评社区感, 共鸣向。红线同公众号(平台克制, 涉投资/敏感必拦)。"""
    title = _title(text)
    for w in WECHAT_REDLINES:
        if w in text:
            return f"[红线拦截] 命中微信读书红线词: {w}。改写后再发。"
    lines = _body_lines(text)
    # 微信读书读者吃"读完这本书/这个领域我想到的事"的共鸣口吻, 长文正文保留, 头尾加社区感导语
    head = f"{title}\n\n大伙在微信读书刷到这篇, 说明咱们关心同一件事。\n"
    body = re.sub(r"^#\s+.+\n?", "", text.strip(), count=1)  # 正文去重标题
    tail = ("\n\n---\n如果哪一段戳到你, 微信读书里划线标记一下, 咱们评论区接着聊。")
    return head + body + tail


def to_weread_idea(text, max_len=480):
    """微信读书想法: 短帖(≤480字), 从长文抽一个最有共鸣的点, 呼吸感短行。"""
    lines = _body_lines(text)
    if len(lines) < 4:
        return "[跳过] 正文太短, 抽不出想法。"
    # 取中间偏后的一段(观点密度高) + 一个数字锚点行
    pick = lines[len(lines) // 3: len(lines) // 3 + 6]
    body = "\n".join(l[:60] for l in pick)
    if len(body) > max_len:
        body = body[:max_len - 1] + "…"
    return f"{body}\n\n你怎么看, 评论区聊聊。"


def to_binance(text):
    """币安广场: 只收区块链内容, 非加密主题直接拒; 加密主题做观察视角短文。"""
    if not any(k in text for k in CRYPTO_HINTS):
        return "[跳过] 非区块链内容, 币安广场不发(路由矩阵: crypto only)。"
    hook = _title(text)
    lines = _body_lines(text, limit=12)
    return (f"{hook}\n\n" + "\n".join(lines) +
            "\n\n观察视角分享, 不构成任何投资建议, 不涉个人持仓。")


def detect_type(text):
    """内容类型识别: 命中加密关键词→crypto, 否则→tech"""
    return "crypto" if any(k in text for k in CRYPTO_HINTS) else "tech"


def adapt_all(text, voice=""):
    """返回 {platform: content}。类型识别 + 双 X 账号路由 + 红线过滤。
    tech:  知乎/小红书/X技术号/领英/公众号/微信读书
    crypto: X区块链号/币安广场
    """
    ctype = detect_type(text)
    outs = {}
    if ctype == "tech":
        outs["x_tech"] = to_x(text)
        outs["zhihu"] = to_zhihu(text)
        outs["xhs"] = to_xhs(text)
        outs["linkedin"] = to_linkedin(text)
        outs["wechat"] = to_wechat(text)      # 可能红线拦截
        outs["weread"] = to_weread(text)
    else:  # crypto
        outs["x_crypto"] = to_x(text)         # X 区块链号: 线程
        outs["binance"] = to_binance(text)    # 非 crypto 会被拒
    return {k: v for k, v in outs.items() if not v.startswith(("[红线拦截]", "[跳过]"))}
