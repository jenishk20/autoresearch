"""Report narrative (main plan §8.4). report.json is built by code; the LLM may only write prose that
uses numbers from it. Every number in the prose is checked against the JSON; failures -> regenerate
once -> templated fallback."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, conlist

from control.llm import LLMFailed


class Narrative(BaseModel):
    headline: str
    summary: str
    findings: conlist(str, min_length=1, max_length=6)
    caveats: list[str] = []


SYSTEM = """You write the narrative of an ML research report. Use ONLY numbers that appear in the given report JSON
(you may round them as they appear there, to at most 1 decimal). Do not compute new numbers. Answer the user's
questions directly. Say "tie" when confidence intervals overlap. Mention which predictions held.
Schema: {"headline": str, "summary": str, "findings": [str], "caveats": [str]}"""

_NUM = re.compile(r"(?<![A-Za-z_])-?\d[\d,]*(?:\.\d+)?")
_STRIP = re.compile(r"(HalfCheetah|Walker2d|Hopper|InvertedDoublePendulum)-v\d|\b\d+x\d+\b|\bP\d+\b|\b[0-9a-f]{12,64}\b")


def _numbers_in_json(obj, acc):
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in_json(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in_json(v, acc)
    elif isinstance(obj, str):
        for m in _NUM.findall(_STRIP.sub(" ", obj)):
            try:
                acc.append(float(m.replace(",", "")))
            except ValueError:
                pass
    return acc


def check_numbers(text: str, report: dict) -> list[str]:
    known = _numbers_in_json(report, [])
    bad = []
    for tok in _NUM.findall(_STRIP.sub(" ", text)):
        v = float(tok.replace(",", ""))
        if abs(v) <= 10 and float(v).is_integer():
            continue  # small counts ("3 seeds", "2 envs")
        dec = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.51 * 10 ** (-dec) + 1e-9
        ok = any(abs(v - k) <= tol or abs(v - k * 100) <= tol or abs(v - round(k)) <= tol and dec == 0 for k in known)
        if not ok:
            bad.append(tok)
    return bad


def template(report: dict) -> Narrative:
    findings = []
    for env, t in report["per_env"].items():
        w = t.get("winner")
        if w:
            row = next(r for r in t["table"] if r["config"] == w)
            tie = " (statistical tie with the runner-up)" if t.get("tie") else ""
            findings.append(f"{env}: {w} wins with IQM {row['iqm']} (95% CI {row['ci'][0]} to {row['ci'][1]}, "
                            f"{row['n']} seeds){tie}.")
    pr = report["predictions"]
    held = [p["id"] for p in pr if p["outcome"] == "confirmed"]
    failed = [p["id"] for p in pr if p["outcome"] == "refuted"]
    findings.append(f"Predictions confirmed: {', '.join(held) or 'none'}; refuted: {', '.join(failed) or 'none'}; "
                    f"the rest were inconclusive.")
    if report.get("halving"):
        findings.append(f"Successive halving saved {report['halving']['saved_pct']}% of screening steps.")
    first = findings[0] if findings else "No verified results."
    return Narrative(headline=first, summary=report["task_summary"], findings=findings,
                     caveats=["Few seeds per config; confidence intervals are wide.",
                              "Narrative generated from a template (no LLM)."])


def narrate(llm, session_id: str, report: dict) -> tuple[Narrative, str, list[str]]:
    compact = json.dumps({k: v for k, v in report.items() if k not in ("decisions", "artifacts", "events")})[:14000]
    user = f"User query: {report['query']}\nReport JSON:\n{compact}"
    for _ in range(2):
        try:
            n = llm.ask_json("report", session_id, SYSTEM, user, Narrative, temperature=0.3)
        except LLMFailed:
            break
        text = " ".join([n.headline, n.summary, *n.findings, *n.caveats])
        bad = check_numbers(text, report)
        if not bad:
            return n, "llm", []
        user += f"\n\nThese numbers are not in the JSON: {bad}. Rewrite using only JSON numbers."
    return template(report), "template", []
