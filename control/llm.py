"""Vultr Serverless Inference client. No tool calling: prompt -> JSON -> pydantic -> bounded action.
Every call is logged to `events` (kind=llm_call). Callers must have a rule-based fallback."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import httpx
from openai import OpenAI
from pydantic import BaseModel, ValidationError


class LLMFailed(Exception):
    pass


def extract_first_json_object(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in model output")


class LLM:
    def __init__(self, cfg: dict, log_event):
        self.cfg = cfg
        self.log_event = log_event  # (session_id, kind, payload) -> None
        self.key = os.environ.get("VULTR_INFERENCE_API_KEY", "")
        self.client = OpenAI(base_url=cfg["base_url"], api_key=self.key or "missing", max_retries=0)
        self.model = self._pick_model()
        self.disabled_reason: str | None = None if self.key else "no VULTR_INFERENCE_API_KEY"
        self.calls, self.seconds, self.failures = 0, 0.0, 0

    def _pick_model(self) -> str:
        wanted = os.environ.get("LLM_MODEL") or self.cfg["model"]
        try:
            r = httpx.get(self.cfg["base_url"].rstrip("/") + "/models", timeout=15)
            ids = {m["id"] for m in r.json().get("data", [])}
        except Exception:
            return wanted
        for m in [wanted, self.cfg["model"], *self.cfg.get("fallback_models", [])]:
            if m in ids:
                return m
        return wanted

    def probe(self):
        """One tiny call at startup so the UI shows whether the key actually works."""
        if self.disabled_reason:
            return
        try:
            self.client.chat.completions.create(model=self.model, max_tokens=1, timeout=20,
                                                messages=[{"role": "user", "content": "ping"}])
        except Exception as e:
            if "401" in str(e) or "Invalid API key" in str(e):
                self.disabled_reason = "Vultr rejected the API key (401) - using rule-based fallbacks"
            else:
                self.disabled_reason = None  # transient errors: keep trying per call

    @property
    def status(self) -> dict:
        return {"model": self.model, "enabled": self.disabled_reason is None, "reason": self.disabled_reason,
                "calls": self.calls, "seconds": round(self.seconds, 1), "failures": self.failures}

    def ask_json(self, stage: str, session_id: str | None, system: str, user: str,
                 schema: type[BaseModel], temperature: float = 0.2, retries: int = 1):
        if self.disabled_reason:
            raise LLMFailed(self.disabled_reason)
        msgs = [{"role": "system", "content": system + "\n\nReturn ONLY one JSON object matching the schema. "
                 "No prose, no markdown fences."},
                {"role": "user", "content": user}]
        last_err = None
        for attempt in range(retries + 1):
            t0 = time.time()
            payload = {"stage": stage, "model": self.model, "attempt": attempt,
                       "prompt_sha": hashlib.sha256((system + user).encode()).hexdigest()[:12]}
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=msgs, temperature=temperature,
                    timeout=self.cfg.get("timeout_s", 60), max_tokens=self.cfg.get("max_tokens", 2500))
                txt = r.choices[0].message.content or ""
                dt = time.time() - t0
                self.calls += 1
                self.seconds += dt
                payload.update(latency_s=round(dt, 2), tokens=getattr(r.usage, "total_tokens", None))
            except Exception as e:  # network / auth / timeout
                dt = time.time() - t0
                self.failures += 1
                msg = str(e)
                if "401" in msg or "Invalid API key" in msg:
                    self.disabled_reason = "Vultr rejected the API key (401) - using rule-based fallbacks"
                payload.update(ok=False, error=msg[:300], latency_s=round(dt, 2))
                self.log_event(session_id, "llm_call", payload)
                raise LLMFailed(msg) from e
            try:
                out = schema.model_validate_json(extract_first_json_object(txt))
                payload.update(ok=True)
                self.log_event(session_id, "llm_call", payload)
                return out
            except (ValidationError, ValueError) as e:
                last_err = e
                self.failures += 1
                payload.update(ok=False, error=f"invalid output: {str(e)[:300]}")
                self.log_event(session_id, "llm_call", payload)
                msgs += [{"role": "assistant", "content": txt[:6000]},
                         {"role": "user", "content": f"Invalid: {str(e)[:1500]}\nFix it and return only the JSON object."}]
        raise LLMFailed(f"invalid JSON after retries: {last_err}")
