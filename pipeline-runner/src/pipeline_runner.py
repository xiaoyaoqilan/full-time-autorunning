#!/usr/bin/env python3
"""可恢复的本地流水线编排器（纯标准库）v1.0

为公司流水线 / 科研论文流水线提供独立于 AI 会话的状态管理：
- 每个角色（stage）独立执行：command 非空则跑 shell 命令，为空则是人工/AI 会话角色（用 done 标记完成）
- 状态与证据落盘 <project>/.pipeline/（state.json + logs/），中断后续跑（running 残留自动重置）
- 质量门禁：human_approval=true 的阶段完成后进入 awaiting_approval，须 approve 显式放行（外部发布/账户操作/新颖性门禁/完整性门禁）
- 失败统一修复机制：command 失败自动重试 retries 次；仍失败则 halt，主管修复后 retry <id> 续跑
- 产物核验：done/命令成功后检查 artifacts 是否存在于 project，缺失视为失败
- 无依赖的 ready 阶段并行执行（reader ∥ scout 之类）

用法:
  python pipeline_runner.py --pipeline pipelines/paper_pipeline.json --project E:\\paper-pipeline\\demo run
  python pipeline_runner.py ... status
  python pipeline_runner.py ... done <stage_id> [--evidence 路径]   # 人工/AI 阶段标记完成
  python pipeline_runner.py ... fail <stage_id> --reason "..."      # 人工阶段标记失败
  python pipeline_runner.py ... approve <stage_id>                  # 门禁放行
  python pipeline_runner.py ... retry <stage_id>                    # 主管修复后续跑
  python pipeline_runner.py ... reset <stage_id>                    # 清零 attempts 强制重跑
  python pipeline_runner.py ... log <stage_id>                      # 查看某阶段执行日志

Pipeline 定义（JSON）字段:
  name / default_retries / max_workers
  stages[]: id, title, role, command(可空=人工), artifacts[], depends_on[],
            human_approval(bool), optional(bool), retries(可覆盖)
"""

import argparse
import datetime
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNNING = "running"
PENDING = "pending"
DONE = "done"
FAILED = "failed"
AWAIT = "awaiting_approval"
TERMINAL_BAD = (FAILED,)


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Orchestrator:
    def __init__(self, pipeline_path: str, project: str, state_dir: str = ""):
        self.defn = json.loads(Path(pipeline_path).read_text(encoding="utf-8"))
        self.stages = {s["id"]: s for s in self.defn["stages"]}
        for sid, s in self.stages.items():
            s.setdefault("retries", self.defn.get("default_retries", 2))
            s.setdefault("human_approval", False)
            s.setdefault("optional", False)
            s.setdefault("depends_on", [])
            s.setdefault("artifacts", [])
        self.project = Path(project)
        self.state_dir = Path(state_dir) if state_dir else self.project / ".pipeline"
        self.log_dir = self.state_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.lock = threading.Lock()
        self.halted = False
        self._load()

    # ---------- 状态读写 ----------
    def _load(self):
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            for sid, st in self.state["stages"].items():
                if st["status"] == RUNNING:  # 上次中断残留
                    st["status"] = PENDING
                    st.setdefault("history", []).append(f"{now()} 检测到中断残留，重置为 pending")
        else:
            self.state = {
                "pipeline": self.defn["name"],
                "created": now(),
                "stages": {sid: {"status": PENDING, "attempts": 0, "history": []} for sid in self.stages},
            }
        for sid in self.stages:  # 定义新增阶段时补齐
            self.state["stages"].setdefault(sid, {"status": PENDING, "attempts": 0, "history": []})
        self._save()

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        for i in range(6):  # Windows 并发下 os.replace 偶发 WinError 5，退避重试
            try:
                tmp.replace(self.state_path)
                return
            except PermissionError:
                time.sleep(0.05 * (i + 1))
        tmp.replace(self.state_path)

    def _set(self, sid: str, status: str, note: str = "", **extra):
        with self.lock:
            st = self.state["stages"][sid]
            st["status"] = status
            st.setdefault("history", []).append(f"{now()} [{status}] {note}".rstrip())
            st.update(extra)
            self._save()

    # ---------- 依赖与调度 ----------
    def _deps_satisfied(self, sid: str) -> bool:
        for dep in self.stages[sid]["depends_on"]:
            st = self.state["stages"][dep]
            if st["status"] == DONE:
                continue
            if self.stages[dep]["optional"]:  # 可选阶段未完成不阻塞
                continue
            return False
        return True

    def _ready(self) -> list[str]:
        out = []
        for sid, s in self.stages.items():
            st = self.state["stages"][sid]
            if s["command"] is None:
                continue  # 人工/AI 阶段不进自动执行循环，由 done 命令标记
            if st["status"] == PENDING and self._deps_satisfied(sid) and not self._gate_blocked(sid):
                out.append(sid)
        return out

    def _gate_blocked(self, sid: str) -> bool:
        for dep in self.stages[sid]["depends_on"]:
            if self.state["stages"][dep]["status"] == AWAIT:
                return True
        return False

    def _check_artifacts(self, sid: str) -> list[str]:
        missing = []
        for art in self.stages[sid]["artifacts"]:
            if not (self.project / art).exists():
                missing.append(art)
        return missing

    # ---------- 执行 ----------
    def _exec(self, sid: str):
        s, st = self.stages[sid], self.state["stages"][sid]
        attempt = st["attempts"] + 1
        self._set(sid, RUNNING, f"第 {attempt} 次执行", attempts=attempt, started=now())
        log_path = self.log_dir / f"{sid}-{attempt}.log"

        if not s["command"]:  # 人工/AI 阶段：由外部 done 标记，不会被调度到
            return

        try:
            proc = subprocess.run(
                s["command"], shell=True, cwd=str(self.project),
                capture_output=True, text=True, timeout=s.get("timeout", 600),
            )
            log_path.write_text(
                f"$ {s['command']}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\nexit={proc.returncode}",
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            log_path.write_text(f"$ {s['command']}\n\n[exception] {e}", encoding="utf-8")
            proc = None

        if proc is None or proc.returncode != 0:
            self._on_failure(sid, attempt, f"命令退出码非 0（日志 {log_path.name}）")
            return
        missing = self._check_artifacts(sid)
        if missing:
            self._on_failure(sid, attempt, f"命令成功但产物缺失: {missing}")
            return
        self._set(sid, AWAIT if s["human_approval"] else DONE,
                  f"成功（第 {attempt} 次）", finished=now(), last_log=str(log_path))
        if s["human_approval"]:
            print(f"[gate] 阶段 {sid} 完成，等待放行：approve {sid}")

    def _on_failure(self, sid: str, attempt: int, reason: str):
        s = self.stages[sid]
        if attempt <= s["retries"]:
            self._set(sid, PENDING, f"失败自动重试（{reason}）", last_fail=reason)
            print(f"[retry] {sid} 第 {attempt} 次失败：{reason}，自动重试")
        else:
            self._set(sid, FAILED, reason, last_fail=reason, finished=now())
            print(f"[halt] {sid} 重试 {s['retries']} 次后仍失败：{reason}，主管修复后 retry {sid}")
            self.halted = True

    def run(self, only: list[str] | None = None):
        print(f"=== {self.defn['name']} 开始编排（断点续跑）===")
        while not self.halted:
            ready = self._ready()
            if only:
                ready = [s for s in ready if s in only]
            if not ready:
                break
            workers = min(self.defn.get("max_workers", 4), len(ready))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(self._exec, ready))
        self.status()
        blocked = [sid for sid in self.stages if self._gate_blocked(sid)]
        if blocked:
            print(f"[blocked] 等门禁放行: {', '.join(blocked)}")
        waiting = [sid for sid, s in self.stages.items()
                   if self.state["stages"][sid]["status"] == PENDING and s["command"] is None and self._deps_satisfied(sid)]
        if waiting:
            print(f"[manual] 等人工/AI 执行后标记: {' '.join(f'done {sid}' for sid in waiting)}")

    # ---------- 人工操作命令 ----------
    def done(self, sid: str, evidence: str = ""):
        self._require(sid)
        missing = self._check_artifacts(sid)
        if missing:
            print(f"[deny] {sid} 产物缺失 {missing}，拒绝标记 done（补齐后再来）")
            sys.exit(2)
        s = self.stages[sid]
        self._set(sid, AWAIT if s["human_approval"] else DONE,
                  f"人工标记完成" + (f"，证据: {evidence}" if evidence else ""), evidence=evidence, finished=now())
        if s["human_approval"]:
            print(f"[gate] {sid} 为门禁阶段，需 approve {sid} 放行下游")

    def fail(self, sid: str, reason: str):
        self._require(sid)
        self._set(sid, FAILED, f"人工标记失败: {reason}", last_fail=reason, finished=now())
        print(f"[failed] {sid} 已标记失败，修复后 retry {sid}")

    def approve(self, sid: str):
        self._require(sid)
        if self.state["stages"][sid]["status"] != AWAIT:
            print(f"[deny] {sid} 当前状态 {self.state['stages'][sid]['status']}，不是 awaiting_approval")
            sys.exit(2)
        self._set(sid, DONE, "门禁放行", approved=now())
        print(f"[approved] {sid} 放行，下游解锁")

    def retry(self, sid: str):
        self._require(sid)
        st = self.state["stages"][sid]
        if st["status"] not in (FAILED, PENDING):
            print(f"[deny] {sid} 状态 {st['status']} 无需 retry")
            sys.exit(2)
        self._set(sid, PENDING, "主管修复后重新入队")
        self.run(only=[sid] if st["attempts"] > 0 else None)

    def reset(self, sid: str):
        self._require(sid)
        self._set(sid, PENDING, "强制重置（attempts 清零）", attempts=0, last_fail="")
        print(f"[reset] {sid} 已重置")

    def log(self, sid: str):
        st = self.state["stages"][sid]
        lp = st.get("last_log")
        if lp and Path(lp).exists():
            print(Path(lp).read_text(encoding="utf-8")[-3000:])
        else:
            print("\n".join(st.get("history", [])) or "(无日志)")

    def _require(self, sid: str):
        if sid not in self.stages:
            print(f"[error] 未知阶段 {sid}，可用: {', '.join(self.stages)}")
            sys.exit(2)

    def status(self):
        print(f"\n=== {self.defn['name']} @ {self.project} ===")
        order = {sid: i for i, sid in enumerate(self.stages)}
        for sid in sorted(self.state["stages"], key=order.get):
            s, st = self.stages[sid], self.state["stages"][sid]
            flags = []
            if s["human_approval"]:
                flags.append("门禁")
            if s["optional"]:
                flags.append("可选")
            if s["command"] is None:
                flags.append("人工")
            print(f"  {st['status']:<18} {sid:<12} {s['title']}"
                  f"{'  [' + ','.join(flags) + ']' if flags else ''}"
                  f"{'  ← ' + st['last_fail'] if st.get('last_fail') and st['status'] == FAILED else ''}")
        print()


def main():
    ap = argparse.ArgumentParser(description="可恢复的本地流水线编排器")
    ap.add_argument("--pipeline", required=True, help="流水线定义 JSON 路径")
    ap.add_argument("--project", required=True, help="项目目录（状态落盘在其 .pipeline/）")
    ap.add_argument("--state-dir", default="", help="覆盖状态目录（默认 <project>/.pipeline）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("stages", nargs="*")
    sub.add_parser("status")
    d = sub.add_parser("done"); d.add_argument("stage"); d.add_argument("--evidence", default="")
    f = sub.add_parser("fail"); f.add_argument("stage"); f.add_argument("--reason", default="人工标记失败")
    a = sub.add_parser("approve"); a.add_argument("stage")
    t = sub.add_parser("retry"); t.add_argument("stage")
    s = sub.add_parser("reset"); s.add_argument("stage")
    l = sub.add_parser("log"); l.add_argument("stage")
    args = ap.parse_args()

    orch = Orchestrator(args.pipeline, args.project, args.state_dir or None)
    if args.cmd == "run":
        orch.run(args.stages or None)
    elif args.cmd == "status":
        orch.status()
    elif args.cmd == "done":
        orch.done(args.stage, args.evidence)
    elif args.cmd == "fail":
        orch.fail(args.stage, args.reason)
    elif args.cmd == "approve":
        orch.approve(args.stage)
    elif args.cmd == "retry":
        orch.retry(args.stage)
    elif args.cmd == "reset":
        orch.reset(args.stage)
    elif args.cmd == "log":
        orch.log(args.stage)


if __name__ == "__main__":
    main()
