"""Requirements on the OpenAI-compatible runner, with the HTTP layer replaced by a fake."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from finexam10k import build_prompt, load_items, run_openai


def fake_chat(answers: dict[str, str], finish: str = "stop"):
    by_prompt = {build_prompt(item): item for item in load_items()}

    def chat(base_url, api_key, body, **_):
        item = by_prompt[body["messages"][0]["content"]]
        return {"choices": [{"message": {"content": json.dumps({"answer": answers.get(item["id"], "A")})},
                             "finish_reason": finish}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    return chat


class RunnerTests(unittest.TestCase):
    def run_main(self, *args: str) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_openai.main(list(args)), 0)

    def test_missing_api_key_is_a_usage_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_openai.main(["--model", "m", "--output", "x.jsonl"])

    def test_writes_one_record_per_item_and_resumes(self) -> None:
        items = load_items()[:5]
        gold = {i["id"]: i["answer"] for i in items}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            out = Path(tmp) / "run.jsonl"
            with mock.patch.object(run_openai, "chat", side_effect=fake_chat(gold)) as chat:
                self.run_main("--model", "m", "--output", str(out), "--limit", "3")
                self.run_main("--model", "m", "--output", str(out), "--limit", "5")
                self.assertEqual(chat.call_count, 5)
                body = chat.call_args.args[2]
                self.assertEqual((body["max_tokens"], body["temperature"]), (8192, 0.0))
            records = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual([r["id"] for r in records], [i["id"] for i in items])
        self.assertEqual([r["prediction"] for r in records], [gold[i["id"]] for i in items])
        self.assertFalse(any("test" in json.dumps(r) for r in records))  # the API key is never written

    def test_failed_items_are_retried_on_rerun(self) -> None:
        items = load_items()[:3]
        gold = {i["id"]: i["answer"] for i in items}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            out = Path(tmp) / "run.jsonl"
            with mock.patch.object(run_openai, "chat", side_effect=RuntimeError("down")):
                self.run_main("--model", "m", "--output", str(out), "--limit", "3")
            with mock.patch.object(run_openai, "chat", side_effect=fake_chat(gold)) as chat:
                self.run_main("--model", "m", "--output", str(out), "--limit", "3")
                self.assertEqual(chat.call_count, 3)
            records = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(records), 6)
        self.assertEqual([r["prediction"] for r in records[3:]], [gold[i["id"]] for i in items])

    def test_truncated_outputs_and_request_errors_count_as_unanswered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            out = Path(tmp) / "run.jsonl"
            with mock.patch.object(run_openai, "chat", side_effect=fake_chat({}, finish="length")):
                self.run_main("--model", "m", "--output", str(out), "--limit", "2", "--no-sampling-params")
            with mock.patch.object(run_openai, "chat", side_effect=RuntimeError("boom")):
                self.run_main("--model", "m", "--output", str(Path(tmp) / "err.jsonl"), "--limit", "1")
            truncated = [json.loads(line) for line in out.read_text().splitlines()]
            failed = json.loads((Path(tmp) / "err.jsonl").read_text())
        self.assertEqual([r["prediction"] for r in truncated], [None, None])
        self.assertEqual((failed["prediction"], failed["error"].startswith("RuntimeError")), (None, True))


if __name__ == "__main__":
    unittest.main()
