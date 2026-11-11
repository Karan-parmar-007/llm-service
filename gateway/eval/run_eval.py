"""Check how the formatter handles spoken self-corrections.

Correction cases must keep only the final version; "keep-*" cases must not lose
anything. Runs the real /format route in-process against Ollama (default), or
against a running gateway with --url.

  python eval/run_eval.py                          # in-process, OLLAMA_BASE/OLLAMA_MODEL from env
  python eval/run_eval.py --url http://127.0.0.1:8003 --token "$FLOW_TOKEN"
  python eval/run_eval.py --runs 3                 # repeat each case (the model is sampled)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def check(case: dict, output: str) -> list[str]:
    out = _norm(output)
    problems = []
    for options in case["must_have"]:
        if not any(re.search(rf"(?<![\w]){re.escape(o.lower())}(?![\w])", out) for o in options):
            problems.append(f"missing {options}")
    for bad in case["must_not"]:
        if re.search(rf"(?<![\w]){re.escape(bad.lower())}(?![\w])", out):
            problems.append(f"still has '{bad}'")
    return problems


def formatter(url: str | None, token: str):
    if url:
        import httpx

        headers = {"X-Flow-Token": token}

        def run(text: str) -> str:
            r = httpx.post(f"{url.rstrip('/')}/format", json={"text": text, "script": "roman"},
                           headers=headers, timeout=180)
            body = r.json()
            while r.status_code == 202:
                time.sleep(0.5)
                r = httpx.get(f"{url.rstrip('/')}/jobs/{body['id']}", headers=headers, timeout=60)
                body = r.json()
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {body}")
            return (body.get("result") or body).get("text", "")
        return run

    os.environ.setdefault("LLM_ENABLED", "1")
    os.environ["SPEECH_ENABLED"] = "0"
    os.environ["FLOW_TOKEN"] = ""
    os.environ["KEEPALIVE_ENABLED"] = "0"
    sys.path.insert(0, str(HERE.parent))
    import app  # noqa: E402

    client = app.app.test_client()

    def run(text: str) -> str:
        r = client.post("/format", json={"text": text, "script": "roman"})
        body = r.get_json()
        while r.status_code == 202:  # long calls become a job to poll, like the real clients
            time.sleep(0.5)
            r = client.get(f"/jobs/{body['id']}")
            body = r.get_json()
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {body}")
        return (body.get("result") or body).get("text", "")
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--token", default=os.environ.get("FLOW_TOKEN", ""))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--show", action="store_true", help="print every output")
    args = ap.parse_args()

    cases = json.loads((HERE / "corrections.json").read_text())
    run = formatter(args.url, args.token)
    passed = total = 0
    started = time.time()
    for case in cases:
        for _ in range(args.runs):
            total += 1
            out = run(case["text"])
            problems = check(case, out)
            ok = not problems
            passed += ok
            if args.show or not ok:
                mark = "PASS" if ok else "FAIL"
                print(f"{mark} {case['id']:<22} -> {out!r}" + ("" if ok else f"   [{'; '.join(problems)}]"))
    fixes = [c for c in cases if not c["id"].startswith("keep-")]
    print(f"\n{passed}/{total} passed  ({len(fixes)} correction cases, {len(cases) - len(fixes)} must-keep cases, "
          f"{args.runs} run(s) each, {time.time() - started:.0f}s)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
