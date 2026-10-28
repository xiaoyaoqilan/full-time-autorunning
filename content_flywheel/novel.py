#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
novel.py · 网文连载引擎（日更一章状态机）
结构:
  novel/bible.md      人设/世界观/分卷大纲/文风契约（唯一事实源）
  novel/state.json    章节计数与断点记录
  novel/chapters/     第NNN章_标题.md
用法:
  python novel.py --status                    # 看进度
  python novel.py --init "题材/一句话设定"     # LLM 生成 bible(需 .env)
  python novel.py --gen                       # 写下一章(读 bible+上一章续写)
  python novel.py --gen 3                     # 连写 3 章
  python novel.py --bible                     # 打印当前 bible 路径, 手工编辑用
"""
import argparse, datetime, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent
NOVEL = ROOT / "novel"
CHAPTERS = NOVEL / "chapters"
BIBLE = NOVEL / "bible.md"
STATE = NOVEL / "state.json"

CHAPTER_WORDS = 3000   # 单章目标字数
CONTEXT_CHARS = 3500   # 续写时带上一章尾部多少字


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"book": "未命名", "chapter": 0, "log": []}


def save_state(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pipeline():
    """复用主 pipeline 的 LLM 调用与 voice 读取"""
    sys.path.insert(0, str(ROOT))
    import pipeline
    return pipeline


def last_chapter_text():
    files = sorted(CHAPTERS.glob("第*.md"))
    if not files:
        return ""
    return files[-1].read_text(encoding="utf-8")


def novel_prompt(bible, prev_tail, chapter_no):
    return (
        f"你是网文连载引擎。按 bible 继续写第 {chapter_no} 章。\n\n"
        f"=== BIBLE(唯一事实源, 人设/世界观/伏笔不得违背) ===\n{bible}\n\n"
        f"=== 上一章结尾(无缝衔接, 不得复述) ===\n{prev_tail}\n\n"
        f"硬性要求:\n"
        f"1. 单章 {CHAPTER_WORDS} 字左右, 小标题格式: 第{chapter_no}章 标题\n"
        f"2. 开头 3 行内必须有钩子(承接上一章悬念或新冲突)\n"
        f"3. 中段至少一个爽点/反转, 对白推动剧情, 不写大段静态描写\n"
        f"4. 章末必须留钩子(悬念/危机/新信息), 禁止本章内收口全部矛盾\n"
        f"5. 严格遵守 bible 的文风契约与人设口癖; 不得出现 bible 外的新设定\n"
        f"6. 直接输出正文, 不要任何解释、大纲或作者的话\n"
    )


def cmd_init(args):
    p = load_pipeline()
    env = p.load_env()
    if not (env.get("LLM_BASE_URL") and env.get("LLM_API_KEY")):
        print("[novel-init] 缺 LLM 配置(.env), 无法生成 bible。可手写 novel/bible.md, 格式见 bible 模板注释。")
        return
    bible = p.call_llm(
        f"为网文连载设计一本新书。题材/一句话设定: {args.init}\n\n"
        "输出 bible.md, 必须包含以下章节:\n"
        "# 书名\n# 一句话卖点\n# 主角(人设/口癖/目标/缺陷)\n# 世界观(核心规则, 少而硬)\n"
        "# 金手指/核心爽点机制\n# 分卷大纲(第一卷至少 30 章的一句话梗概, 标注每章章末钩子)\n"
        "# 文风契约(节奏/对白比例/禁用词/叙述人称)\n# 伏笔账本(埋设章=>回收章)\n\n"
        "要求: 卖点必须一句话能让人想点进去; 大纲每章都要有钩子; 文风契约写明可执行的规则。")
    if bible:
        NOVEL.mkdir(exist_ok=True)
        CHAPTERS.mkdir(exist_ok=True)
        BIBLE.write_text(bible, encoding="utf-8")
        st = load_state()
        m = re.search(r"#\s*书名\s*\n+(.+)", bible)
        st["book"] = m.group(1).strip() if m else "未命名"
        save_state(st)
        print(f"[novel-init] bible 已生成: {BIBLE}\n[novel-init] 书名: {st['book']}\n"
              "[novel-init] 强烈建议人工过一遍 bible 再开更(改到满意为止, 开更后尽量别改设定)")


def cmd_gen(args):
    if not BIBLE.exists():
        print("[novel] 还没有 novel/bible.md。先跑 --init 或手写。")
        return
    p = load_pipeline()
    n = args.gen if args.gen and args.gen > 0 else 1
    st = load_state()
    bible = BIBLE.read_text(encoding="utf-8")
    CHAPTERS.mkdir(exist_ok=True)
    for _ in range(n):
        no = st["chapter"] + 1
        prev = last_chapter_text()
        prev_tail = prev[-CONTEXT_CHARS:] if prev else "(本书第一章, 无上一章)"
        print(f"[novel] 写第 {no} 章 ...")
        art = p.call_llm(novel_prompt(bible, prev_tail, no), temperature=0.85)
        if not art:
            print("[novel] LLM 未返回, 停止(检查 .env)")
            return
        m = re.match(r"#?\s*第\s*0*(\d+)\s*章\s*(.*)", art.strip().splitlines()[0])
        title = m.group(2).strip() if m else "未命名"
        fn = CHAPTERS / f"第{no:03d}章_{re.sub(r'[\\\\/:*?\"<>|]', '_', title)[:20]}.md"
        if list(CHAPTERS.glob(f'第{no:03d}章_*.md')):
            raise FileExistsError(f'第 {no} 章已存在；请先核对 state.json，避免重复写入')
        fn.write_text(art.strip() + "\n", encoding="utf-8")
        st["chapter"] = no
        st["log"].append({"chapter": no, "title": title, "file": fn.name,
                          "date": str(datetime.date.today())})
        save_state(st)
        print(f"[novel] 完成: {fn.name}")
    print(f"[novel] 当前进度: 第 {st['chapter']} 章 / 书名: {st['book']}")


def cmd_status(_):
    st = load_state()
    files = sorted(CHAPTERS.glob("第*.md")) if CHAPTERS.exists() else []
    print(f"书名: {st['book']}")
    print(f"已更: {st['chapter']} 章 / 章节文件: {len(files)} 个")
    bible_flag = "有" if BIBLE.exists() else "无(先 --init)"
    print(f"bible: {bible_flag}")
    for e in st.get("log", [])[-5:]:
        print(f"  第{e['chapter']:03d}章 {e['title']}  ({e['date']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="novel serial engine")
    ap.add_argument("--init", metavar="设定")
    ap.add_argument("--gen", nargs="?", const=1, type=int, default=None)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.init:
        cmd_init(a)
    elif a.gen is not None:
        a.gen = a.gen  # argparse nargs 给出章数
        cmd_gen(a)
    elif a.status:
        cmd_status(a)
    else:
        cmd_status(a)
