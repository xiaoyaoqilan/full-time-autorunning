#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
content_flywheel · 内容飞轮主流水线
用法:
  python pipeline.py --list-topics            # 看话题池待做队列
  python pipeline.py --gen                    # 取下一条选题, 调 LLM 生成长文(呼吸感 voice)入库
  python pipeline.py --adapt <文章文件名>      # 一篇长文 → 各平台成稿 → outbox/<平台>/
  python pipeline.py --adapt-all              # articles/ 里未适配过的全部跑一遍
  python pipeline.py --feedback               # 读取 feedback/metrics.csv, 高流量回流 archive+话题池
  python pipeline.py --push <outbox/x 文件>    # 通过 Buffer 推 X(需 BUFFER_TOKEN; AUTO_PUSH_X=true 才免确认)
  python pipeline.py --refresh-voice          # 重读 voice_samples 更新 voice 规则(需 LLM)
配置: .env (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / BUFFER_TOKEN / AUTO_PUSH_X)
"""
import argparse, csv, datetime, json, os, pathlib, re, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
KB = ROOT / "knowledge_base"
OUT = ROOT / "outbox"
CFG = ROOT / "config"
PLATFORMS = ["zhihu", "xhs", "x_tech", "x_crypto", "linkedin", "wechat", "weread", "binance"]


def load_env():
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def call_llm(prompt, system="", env=None, temperature=0.7):
    """OpenAI 兼容接口(DeepSeek 网关/官方均可)。未配置返回 None。"""
    env = env or load_env()
    base, key = env.get("LLM_BASE_URL", "").rstrip("/"), env.get("LLM_API_KEY", "")
    if not (base and key):
        print("[LLM] 未配置 LLM_BASE_URL / LLM_API_KEY(见 .env.example), 跳过生成")
        return None
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    body = json.dumps({"model": env.get("LLM_MODEL", "gpt-4o"),
                       "messages": msgs, "temperature": temperature}).encode("utf-8")
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def read_voice():
    voice = (CFG / "voice_rules.md").read_text(encoding="utf-8")
    samples = sorted((KB / "voice_samples").glob("*.md")) + sorted((KB / "voice_samples").glob("*.txt"))
    sample_text = ""
    for s in samples[:20]:  # 最多取 20 篇原文防超上下文
        sample_text += f"\n\n--- 样本: {s.name} ---\n" + s.read_text(encoding="utf-8")[:3000]
    return voice, sample_text


def pending_section(pool):
    """只取 '## 待做队列' 到下一个 ## 之间的内容, 避免抓到使用说明示例"""
    m = re.search(r"## 待做队列(.*?)(?=\n## |\Z)", pool, re.S)
    return m.group(1) if m else pool


def next_pending(pool):
    m = re.search(r"- \[ \] (.+)", pending_section(pool))
    return m.group(1).strip() if m else None


def cmd_list_topics(_):
    pool = (KB / "topics" / "topic_pool.md").read_text(encoding="utf-8")
    pending = [l for l in pending_section(pool).splitlines() if l.strip().startswith("- [ ]")]
    print(f"待做选题 {len(pending)} 条:")
    for l in pending:
        print(" ", l.strip())


def cmd_gen(_):
    pool_path = KB / "topics" / "topic_pool.md"
    pool = pool_path.read_text(encoding="utf-8")
    topic_line = next_pending(pool)
    if not topic_line:
        print("[gen] 话题池空了, 先往 topics/topic_pool.md 的待做队列加选题")
        return
    parts = [x.strip() for x in topic_line.split("|")]
    topic, ctype = parts[0], (parts[1] if len(parts) > 1 else "tech")
    hint = parts[2] if len(parts) > 2 else ""
    print(f"[gen] 取题: {topic}  (类型: {ctype})")

    voice, samples = read_voice()
    ref = ""
    refs = sorted((KB / "articles").glob("*.md"))
    if refs:
        ref = refs[-1].read_text(encoding="utf-8")[:4000]  # 最近一篇做结构参照

    system = ("你是'起岚'的技术内容引擎。严格遵守下面的 voice 规则:\n" + voice)
    prompt = (f"写一篇深度技术长文, 选题: {topic}\n角度提示: {hint}\n"
              f"要求: 机制讲到可推导级别, 有数字锚点, 结尾给判断不做总结腔。\n"
              f"结构参照(只学结构不抄内容):\n{ref}\n{samples and ('语气样本(必须模仿其呼吸感):' + samples[:6000])}")
    art = call_llm(prompt, system)
    if not art:
        return
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)[:40]
    out = KB / "articles" / f"{datetime.date.today()}_{slug}.md"
    if out.exists():
        raise FileExistsError(f'文章已存在，拒绝覆盖: {out}')
    out.write_text(f"# {topic}\n\n{art}", encoding="utf-8")
    pool = pool.replace(f"- [ ] {topic_line}",
                        f"- [x] {topic_line}  → {out.name}", 1)
    pool_path.write_text(pool, encoding="utf-8")
    print(f"[gen] 成文: {out}")
    return out


def cmd_adapt(args):
    art_path = KB / "articles" / args.file
    if not art_path.exists():
        art_path = pathlib.Path(args.file)
    if not art_path.exists():
        print(f"[adapt] 找不到文章: {args.file}"); return
    import adapters
    text = art_path.read_text(encoding="utf-8")
    voice, _ = read_voice()
    outs = adapters.adapt_all(text, voice)
    for plat, content in outs.items():
        dest = OUT / plat / art_path.name
        dest.write_text(content, encoding="utf-8")
        print(f"[adapt] {plat:8s} → {dest}")


def cmd_adapt_all(_):
    import adapters
    done_marker = ROOT / ".adapt_done"
    done = set(done_marker.read_text(encoding="utf-8").splitlines()) if done_marker.exists() else set()
    voice, _ = read_voice()
    for art in sorted((KB / "articles").glob("*.md")):
        if art.name in done:
            continue
        outs = adapters.adapt_all(art.read_text(encoding="utf-8"), voice)
        for plat, content in outs.items():
            dest = OUT / plat / art.name
            dest.write_text(content, encoding="utf-8")
            print(f"[adapt] {art.name[:30]:32s} → {plat}")
        done.add(art.name)
    done_marker.write_text("\n".join(sorted(done)), encoding="utf-8")


def cmd_feedback(_):
    """metrics.csv 列: platform,date,title,reads,likes,comments,url"""
    f = ROOT / "feedback" / "metrics.csv"
    if not f.exists():
        f.write_text("platform,date,title,reads,likes,comments,url\n", encoding="utf-8")
        print("[feedback] 已创建 feedback/metrics.csv, 每条发布内容登记一行, reads 达标自动回流")
        return
    threshold = int(load_env().get("FEEDBACK_THRESHOLD", "1000"))
    hits = []
    with open(f, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                if int(row["reads"]) >= threshold:
                    hits.append(row)
            except (ValueError, KeyError):
                continue
    if not hits:
        print(f"[feedback] 暂无 reads>={threshold} 的爆款"); return
    pool_path = KB / "topics" / "topic_pool.md"
    pool = pool_path.read_text(encoding="utf-8")
    today = datetime.date.today()
    for h in hits:
        # 1) 归档: 找到同名文章复制进 archive
        for art in (KB / "articles").glob("*.md"):
            if h["title"] in art.read_text(encoding="utf-8")[:500]:
                (KB / "archive" / art.name).write_text(art.read_text(encoding="utf-8"), encoding="utf-8")
                break
        # 2) 衍生选题追加
        derivative = (f"- [ ] {h['title']} 的续作/反方角度 | "
                      f"{'crypto' if h['platform'] in ('x','binance') else 'tech'} | "
                      f"来源: {h['platform']} reads={h['reads']} 已验证选题, {today} 回流\n")
        identity = f"{h['title']} 的续作/反方角度 |"
        if not any(identity in line for line in pool.splitlines()):
            marker = re.search(r"^## 待做队列\s*$", pool, re.M)
            if not marker:
                raise ValueError('话题池缺少「## 待做队列」，没有写入衍生选题')
            pool = pool[:marker.end()] + "\n" + derivative + pool[marker.end():]
    pool_path.write_text(pool, encoding="utf-8")
    print(f"[feedback] {len(hits)} 条爆款已回流: 归档 archive/ + 追加衍生选题到话题池")


def cmd_push(args):
    print('[push] 已禁用 Buffer 推送；请人工复制 outbox 内容到目标平台。')
    return
    env = load_env()
    token = env.get("BUFFER_TOKEN", "")
    if not token:
        print("[push] 缺 BUFFER_TOKEN(.env), 无法推送"); return
    if env.get("AUTO_PUSH_X", "false").lower() != "true":
        confirm = input("[push] AUTO_PUSH_X=false, 确认推送这条? (y/N) ")
        if confirm.lower() != "y":
            print("[push] 已取消(遵守'说发才发')"); return
    profile = env.get("BUFFER_PROFILE_ID_TECH" if args.account == "tech"
                      else "BUFFER_PROFILE_ID_CRYPTO", "")
    if not profile:
        print(f"[push] 缺 BUFFER_PROFILE_ID_{args.account.upper()}(.env), 无法推送 {args.account} 账号")
        return
    text = pathlib.Path(args.file).read_text(encoding="utf-8")
    body = urllib.parse.urlencode({"access_token": token, "profile_ids[]": profile,
                                   "text": text[:280]}).encode()
    req = urllib.request.Request("https://api.bufferapp.com/1/post/create.json", data=body)
    with urllib.request.urlopen(req, timeout=60) as r:
        print("[push] Buffer 返回:", r.status, r.read()[:200])


def cmd_weread_ideas(args):
    """微信读书想法: 每天几条。有 LLM 用 LLM 生成, 无 LLM 降级为规则抽取(保证全自动不断粮)。"""
    import adapters
    n = args.weread_ideas if args.weread_ideas and args.weread_ideas > 0 else 3
    # 素材源: 最近的 articles + 最近的网文章节
    sources = sorted((KB / "articles").glob("*.md"))[-5:]
    novel_chapters = sorted((ROOT / "novel" / "chapters").glob("第*.md")) if (ROOT / "novel" / "chapters").exists() else []
    sources += novel_chapters[-2:]
    if not sources:
        print("[weread-ideas] 没有素材(articles/ 和 novel/chapters/ 都空), 先 --gen 或 --novel-gen")
        return
    outdir = OUT / "weread_ideas"
    outdir.mkdir(parents=True, exist_ok=True)
    today = str(datetime.date.today())
    made = 0
    voice, samples = read_voice()
    idx = len(list(outdir.glob(f"{today}_*.md")))
    for src in sources:
        if made >= n:
            break
        text = src.read_text(encoding="utf-8")
        idea = None
        env = load_env()
        if env.get("LLM_BASE_URL") and env.get("LLM_API_KEY"):
            idea = call_llm(
                f"从下面内容里提炼一条微信读书'想法'(短帖, 480字内): 只抓一个最有共鸣的点, "
                f"呼吸感短行(一句一行/口语连接词/事实平铺), 无 emoji 无加粗, 不收口不教人, "
                f"称呼读者'大伙', 结尾抛问句出口。零编造, 只用原文里的事实。\n\n"
                f"voice 规则:\n{voice[:2000]}\n\n原文:\n{text[:5000]}", temperature=0.8)
        if not idea:
            idea = adapters.to_weread_idea(text)
        if not idea or idea.startswith("[跳过]"):
            continue
        idx += 1
        dest = outdir / f"{today}_想法{idx:02d}.md"
        dest.write_text(f"(来源: {src.name})\n\n{idea}", encoding="utf-8")
        print(f"[weread-ideas] → {dest}")
        made += 1
    print(f"[weread-ideas] 本次产出 {made} 条(目标 {n})")


def cmd_refresh_voice(_):
    voice, samples = read_voice()
    if not samples:
        print("[voice] voice_samples/ 还是空的, 请先放入你的原文(见 knowledge_base/README.md)"); return
    art = call_llm(
        "以下是 voice 规则和我本人的多篇原文样本。请对比原文, 修订规则文件里不够精确的地方, "
        "输出完整的修订后规则文件(md), 保持原有章节结构, 重点补充: 呼吸节奏细节、口头连接词习惯、"
        "开头/结尾的真实模式、数字锚点的摆放习惯。\n\n规则当前版:\n" + voice + "\n\n样本:\n" + samples[:15000],
        temperature=0.3)
    if art:
        (CFG / "voice_rules.md").write_text(art, encoding="utf-8")
        print("[voice] voice_rules.md 已按样本更新")


def buffer_push(text, account, env):
    """直接推 Buffer。返回 (ok, 信息)"""
    return False, 'Buffer 推送已禁用；请人工发布'
    token = env.get("BUFFER_TOKEN", "")
    pid = env.get("BUFFER_PROFILE_ID_TECH" if account == "tech"
                  else "BUFFER_PROFILE_ID_CRYPTO", "")
    if not token or not pid:
        return False, "缺 BUFFER_TOKEN 或 BUFFER_PROFILE_ID_" + account.upper()
    body = urllib.parse.urlencode({"access_token": token, "profile_ids[]": pid,
                                   "text": text[:280]}).encode()
    req = urllib.request.Request("https://api.bufferapp.com/1/post/create.json", data=body)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return True, f"Buffer {r.status}"
    except Exception as e:
        return False, str(e)


def find_topics_llm(env, n=3):
    """主编模式: LLM 基于 voice 人设 + 已有内容提新选题, 查重后入池"""
    pool = (KB / "topics" / "topic_pool.md").read_text(encoding="utf-8")
    arts = [p.stem for p in (KB / "articles").glob("*.md")]
    voice, _ = read_voice()
    resp = call_llm(
        f"你是内容主编。人设与 voice:\n{voice[:1500]}\n\n"
        f"已写过的文章(可做衍生, 不要重复):\n" + "\n".join(arts[-12:]) +
        f"\n\n已有待做选题(避免重复):\n{pool[:1500]}\n\n"
        f"提出 {n} 条全新选题。要求: 一半 tech 一半 crypto(crypto 必须是观察视角不涉交易); "
        f"选题要蹭时效或有争议性; 每条严格按格式:\n"
        f"- [ ] 题目 | tech或crypto | 一句角度提示\n只输出列表, 不要其他文字。",
        temperature=0.9)
    if not resp:
        return 0
    added = 0
    for line in resp.splitlines():
        line = line.strip()
        if line.startswith("- [ ]") and line not in pool:
            pool = pool.replace("## 衍生选题区",
                                f"{line}\n\n## 衍生选题区", 1)
            added += 1
    (KB / "topics" / "topic_pool.md").write_text(pool, encoding="utf-8")
    return added


def cmd_find_topics(args):
    n = args.find_topics if args.find_topics and args.find_topics > 0 else 3
    added = find_topics_llm(load_env(), n)
    print(f"[find-topics] 新增 {added} 条选题入池")


def quality_gate(art_path, env):
    """LLM 质检: 深度/准确/呼吸感/红线 四项, 返回 (分数或None, 意见)"""
    text = art_path.read_text(encoding="utf-8")
    resp = call_llm(
        "你是质检主编。对下面这篇技术长文打分(0-10, 7 分及格), 维度: 技术准确性/深度/"
        "voice 呼吸感/是否踩红线(交易建议、编造数据、政治宗教)。"
        "只输出一行: `分数: X` 换行 `意见: 一句话`。\n\n" + text[:8000],
        temperature=0.2)
    if not resp:
        return None, "无 LLM, 跳过质检"
    m = re.search(r"分数[:：]\s*(\d+(?:\.\d+)?)", resp)
    note = re.search(r"意见[:：]\s*(.+)", resp)
    return (float(m.group(1)) if m else None), (note.group(1).strip() if note else "")


def cmd_autopilot(args):
    """全自动驾驶: 补题→生成→质检→适配→出待审文件。人工只审这一个文件。"""
    print("=== AUTOPILOT 开始", datetime.datetime.now().strftime("%m-%d %H:%M"), "===")
    env = load_env()
    # 1) 话题保障
    pool = (KB / "topics" / "topic_pool.md").read_text(encoding="utf-8")
    pending_n = len([l for l in pending_section(pool).splitlines()
                     if l.strip().startswith("- [ ]")])
    if pending_n < 3:
        added = find_topics_llm(env, 3)
        print(f"[autopilot] 话题不足, 主编自动补了 {added} 条")
    # 2) 生成
    art = cmd_gen(args)
    if not art:
        print("[autopilot] 生成失败(多半缺 LLM 配置), 终止"); return
    # 3) 质检门
    score, note = quality_gate(art, env)
    def restore_topic():
        pool_path = KB / 'topics' / 'topic_pool.md'
        current_pool = pool_path.read_text(encoding='utf-8')
        current_pool = re.sub(r'^- \[x\] (.+?)  → ' + re.escape(art.name) + r'[ \t]*$', r'- [ ] \1', current_pool, count=1, flags=re.M)
        pool_path.write_text(current_pool, encoding='utf-8')
    if score is not None:
        print(f"[autopilot] 质检: {score}/10 — {note}")
        if score < float(env.get("QUALITY_MIN", "7")):
            rej = ROOT / "review" / "rejected"
            rej.mkdir(parents=True, exist_ok=True)
            if (rej / art.name).exists():
                raise FileExistsError(f'拒绝覆盖已有驳回文件: {rej / art.name}')
            art.rename(rej / art.name)
            restore_topic()
            print(f"[autopilot] 低于 {env.get('QUALITY_MIN','7')} 分, 已驳回进 review/rejected/, 不发布")
            return
    else:
        restore_topic()
        print(f"[autopilot] 质检未完成({note})，停止；文章保留供人工检查，不进入适配")
        return
    # 4) 全平台适配
    import adapters
    outs = adapters.adapt_all(art.read_text(encoding="utf-8"))
    for plat, content in outs.items():
        dest = OUT / plat / art.name
        dest.write_text(content, encoding="utf-8")
        print(f"[autopilot] {plat:9s} → {dest}")
    # 5) 出待审文件(人工唯一动作: 把 status: 待审 改成 ok 或 no)
    ctype = "crypto" if "x_crypto" in outs or "binance" in outs else "tech"
    review = ROOT / "review"
    review.mkdir(exist_ok=True)
    digest = review / f"{datetime.date.today()}_待审.md"
    if digest.exists():
        digest = review / f"{datetime.date.today()}_{art.stem}_待审.md"
    if digest.exists():
        raise FileExistsError(f'拒绝覆盖已有待审文件: {digest}')
    head = (f"# 待审 · {datetime.date.today()}\n"
            f"status: 待审\n"
            f"文章: {art.name}\n"
            f"X账号: {'crypto' if ctype == 'crypto' else 'tech'}\n"
            f"质检: {score if score is not None else '未评'}/10 — {note}\n\n"
            f"审核方法: 审阅正文与证据，需要发布时由人工复制到目标平台。Buffer 不使用。\n\n---\n\n")
    digest.write_text(head + "\n\n".join(f"## [{p}]\n{c}" for p, c in outs.items()),
                      encoding="utf-8")
    print(f"[autopilot] 待审文件: {digest}  ← 人工唯一动作: 改 status → --publish")


def cmd_publish(_):
    """审核收尾: status: ok → X 自动推送; status: no → 驳回; 未审跳过。"""
    print('[publish] 已禁用自动发布和自动清除稿件；请人工审阅并发布。')
    return
    env = load_env()
    review = ROOT / "review"
    review.mkdir(exist_ok=True)
    digests = sorted(review.glob("*_待审.md"))
    if not digests:
        print("[publish] 没有待审文件"); return
    for d in digests:
        text = d.read_text(encoding="utf-8")
        m = re.search(r"^status:\s*(.+)$", text, re.M)
        status = m.group(1).strip() if m else "待审"
        if status == "待审":
            print(f"[publish] {d.name}: 未审核, 跳过"); continue
        art_name = re.search(r"^文章:\s*(.+)$", text, re.M).group(1).strip()
        account = re.search(r"^X账号:\s*(\w+)", text, re.M).group(1).strip()
        art_path = KB / "articles" / art_name
        if status.lower() in ("ok", "y", "是", "通过"):
            xf = OUT / ("x_tech" if account == "tech" else "x_crypto") / art_name
            if xf.exists():
                hook = xf.read_text(encoding="utf-8").split("\n---\n")[0]
                ok, msg = buffer_push(hook, account, env)
                print(f"[publish] X({account}) 推送: {'✅' if ok else '❌'} {msg}")
            # 其余平台稿已在 outbox, 粘贴发布
            print(f"[publish] 其余平台成稿在 outbox/, 打开粘贴即可")
        else:
            rej = ROOT / "review" / "rejected"
            rej.mkdir(exist_ok=True)
            if art_path.exists():
                art_path.rename(rej / art_name)
            for f in ROOT.glob(f"outbox/*/{art_name}"):
                f.unlink()
            print(f"[publish] {art_name} 已驳回, 相关平台稿已清除")
        d.rename(d.with_name(d.name.replace("待审", "已处理")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="content flywheel")
    ap.add_argument("--list-topics", action="store_true")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--adapt", metavar="FILE")
    ap.add_argument("--adapt-all", action="store_true")
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--push", metavar="FILE")
    ap.add_argument("--account", choices=["tech", "crypto"], default="tech",
                    help="X 双账号: tech=技术号, crypto=区块链号")
    ap.add_argument("--weread-ideas", nargs="?", const=3, type=int, default=None,
                    help="生成 N 条微信读书想法(默认 3)")
    ap.add_argument("--refresh-voice", action="store_true")
    ap.add_argument("--autopilot", action="store_true",
                    help="全自动驾驶: 补题→生成→质检→适配→出待审文件")
    ap.add_argument("--find-topics", nargs="?", const=3, type=int, default=None,
                    help="主编模式自动补 N 条选题(默认 3)")
    ap.add_argument("--publish", action="store_true",
                    help="审核收尾: status ok 的自动推 X, no 的驳回")
    a = ap.parse_args()
    if a.adapt:
        a.file = a.adapt; cmd_adapt(a)
    elif a.push:
        a.file = a.push; cmd_push(a)
    elif a.gen: cmd_gen(a)
    elif a.adapt_all: cmd_adapt_all(a)
    elif a.feedback: cmd_feedback(a)
    elif a.weread_ideas is not None: cmd_weread_ideas(a)
    elif a.refresh_voice: cmd_refresh_voice(a)
    elif a.autopilot: cmd_autopilot(a)
    elif a.find_topics is not None: cmd_find_topics(a)
    elif a.publish: cmd_publish(a)
    else: cmd_list_topics(a)
