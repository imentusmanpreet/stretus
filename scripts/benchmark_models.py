#!/usr/bin/env python3
"""
scripts/benchmark_models.py - Send the same prompt to multiple OpenRouter models
and log request/response metrics to CSV.

Run from repo root:

    python3 scripts/benchmark_models.py \\
        --api-key sk-or-v1-... \\
        --models google/gemini-2.5-flash anthropic/claude-3.5-sonnet \\
        --system-prompt "You are Stretus, an algo-trading coach." \\
        --user-prompt "Build an ORB strategy on HDFC Bank using 5m candles."

Output (appended each run):
    benchmark_results/model_runs.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTPUT_CSV = "model_runs.csv"

CSV_FIELDS = [
    "id",
    "batch_id",
    "timestamp_utc",
    "model",
    "status",
    "system_prompt",
    "user_prompt",
    "response",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "temperature",
    "max_tokens",
    "openrouter_id",
    "error_message",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _resolve_prompt(value: str | None, file_path: str | None, label: str) -> str:
    if file_path:
        return _read_text(Path(file_path))
    if value and value.strip():
        return value.strip()
    raise ValueError(f"Provide --{label} or --{label}-file")


async def call_openrouter(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai: pip install openai") from exc

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    started = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    message = resp.choices[0].message
    content = str(getattr(message, "content", "") or "")
    usage = resp.usage

    return {
        "response": content,
        "latency_ms": latency_ms,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "openrouter_id": getattr(resp, "id", None),
    }


def append_rows(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUTPUT_CSV
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return path


async def run(args: argparse.Namespace) -> int:
    system_prompt = _resolve_prompt(args.system_prompt, args.system_prompt_file, "system-prompt")
    user_prompt = _resolve_prompt(args.user_prompt, args.user_prompt_file, "user-prompt")
    output_dir = Path(args.output_dir)
    batch_id = str(uuid.uuid4())
    rows: list[dict[str, Any]] = []

    print(f"Batch: {batch_id}")
    print(f"Models: {', '.join(args.models)}")
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"User prompt: {len(user_prompt)} chars\n")

    for model in args.models:
        row: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "batch_id": batch_id,
            "timestamp_utc": _utc_now(),
            "model": model,
            "status": "success",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": "",
            "latency_ms": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "openrouter_id": "",
            "error_message": "",
        }

        try:
            result = await call_openrouter(
                api_key=args.api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            row.update({
                "response": result["response"],
                "latency_ms": result["latency_ms"],
                "prompt_tokens": result["prompt_tokens"] or "",
                "completion_tokens": result["completion_tokens"] or "",
                "total_tokens": result["total_tokens"] or "",
                "openrouter_id": result["openrouter_id"] or "",
            })
            print(
                f"[ok] {model} | {row['latency_ms']}ms | "
                f"tokens={row['total_tokens'] or 'n/a'}"
            )
        except Exception as exc:
            row["status"] = "error"
            row["error_message"] = str(exc)[:500]
            print(f"[error] {model} | {exc}")

        rows.append(row)

        if args.delay_seconds > 0:
            await asyncio.sleep(args.delay_seconds)

    csv_path = append_rows(output_dir, rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the same prompt across multiple OpenRouter models and log results to CSV.",
    )
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--models", nargs="+", required=True, help="OpenRouter model IDs")
    parser.add_argument("--system-prompt", default=None, help="System prompt text")
    parser.add_argument("--system-prompt-file", default=None, help="Path to system prompt file")
    parser.add_argument("--user-prompt", default=None, help="User prompt text")
    parser.add_argument("--user-prompt-file", default=None, help="Path to user prompt file")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for CSV output (default: benchmark_results)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Pause between model calls (default: 0.5)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
