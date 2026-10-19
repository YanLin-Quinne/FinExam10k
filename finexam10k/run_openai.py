"""Run a model on the public partition through any OpenAI-compatible chat completions endpoint.

The API key is read from an environment variable and never written to disk. A rerun skips items that already have
a successful record and retries the ones that failed.

    export OPENAI_API_KEY=...
    python -m finexam10k.run_openai --model gpt-4o --output runs/gpt-4o.jsonl --limit 20
    python -m finexam10k.evaluate runs/gpt-4o.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .data import DEFAULT_PATH, load_items
from .prompt import messages
from .scoring import parse_answer


def chat(base_url: str, api_key: str, body: dict, *, retries: int = 3, timeout: float = 600) -> dict:
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--track", choices=("all", "context-complete"), default="all")
    parser.add_argument("--limit", type=int, help="only the first N items, for a quick check")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0, help="omitted for --no-sampling-params models")
    parser.add_argument("--no-sampling-params", action="store_true", help="for endpoints that reject temperature")
    parser.add_argument("--json-mode", action="store_true", help="request response_format json_object")
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH, help="path to finexam10k_public_5110.jsonl")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"set the {args.api_key_env} environment variable")
    items = load_items(args.data, context_complete=True if args.track == "context-complete" else None)[:args.limit]
    done = set()
    if args.output.exists():
        records = (json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
        done = {record["id"] for record in records if not record.get("error")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    todo = [item for item in items if item["id"] not in done]
    print(f"{len(items)} items, {len(done)} already done, {len(todo)} to run")
    with args.output.open("a", encoding="utf-8") as sink:
        for count, item in enumerate(todo, 1):
            body = {"model": args.model, "messages": messages(item), "max_tokens": args.max_tokens}
            if not args.no_sampling_params:
                body["temperature"] = args.temperature
            if args.json_mode:
                body["response_format"] = {"type": "json_object"}
            record = {"id": item["id"], "model": args.model}
            try:
                reply = chat(args.base_url, api_key, body)
                choice = reply["choices"][0]
                record.update(output=choice["message"].get("content") or "", finish_reason=choice.get("finish_reason"),
                              usage=reply.get("usage"))
                # length-truncated outputs count as incorrect, as in the paper
                record["prediction"] = None if choice.get("finish_reason") == "length" else parse_answer(record["output"])
            except Exception as error:  # recorded so the run can continue, and a rerun retries this item
                record.update(output="", prediction=None, error=f"{type(error).__name__}: {error}"[:300])
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            if count % 50 == 0 or count == len(todo):
                print(f"{count}/{len(todo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
