import argparse
import asyncio
import json
import os
import shlex
import sys
from typing import Any, Dict, List
from urllib.parse import urlsplit

import httpx

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.core.ollama_runtime import candidate_ollama_base_urls


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _normalize_base_url(value: str) -> str:
    cleaned = (value or "").strip().rstrip("/")
    if not cleaned:
        return "http://127.0.0.1:11434"
    if "://" not in cleaned:
        cleaned = f"http://{cleaned}"
    return cleaned


def _merge_candidates(configured_base_url: str) -> List[str]:
    merged: List[str] = []
    seen = set()
    for source in (
        candidate_ollama_base_urls(configured_base_url),
        candidate_ollama_base_urls("http://127.0.0.1:11434"),
    ):
        for url in source:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


async def probe_endpoint(base_url: str, timeout: httpx.Timeout) -> Dict[str, Any]:
    endpoint = f"{base_url}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", []) if isinstance(payload, dict) else []
            return {
                "base_url": base_url,
                "ok": True,
                "status_code": response.status_code,
                "model_count": len(models) if isinstance(models, list) else 0,
                "error": "",
            }
    except Exception as exc:
        return {
            "base_url": base_url,
            "ok": False,
            "status_code": None,
            "model_count": 0,
            "error": _safe_str(exc, repr(exc)) or repr(exc),
        }


async def discover_reachable_endpoint(
    configured_base_url: str, connect_timeout: float, total_timeout: float
) -> Dict[str, Any]:
    configured = _normalize_base_url(configured_base_url)
    candidates = _merge_candidates(configured)
    timeout = httpx.Timeout(total_timeout, connect=connect_timeout)

    attempts: List[Dict[str, Any]] = []
    for candidate in candidates:
        result = await probe_endpoint(candidate, timeout)
        attempts.append(result)
        if result["ok"]:
            return {
                "configured_base_url": configured,
                "reachable_base_url": candidate,
                "attempts": attempts,
            }

    return {
        "configured_base_url": configured,
        "reachable_base_url": "",
        "attempts": attempts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Ollama /api/tags candidates and print the first reachable "
            "OLLAMA_BASE_URL for this runtime (useful for WSL -> host discovery)."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Preferred Ollama base URL used to derive fallback candidates",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=1.5,
        help="Per-endpoint connect timeout in seconds",
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=3.0,
        help="Per-endpoint total timeout in seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result instead of text output",
    )
    return parser


def _print_text_summary(result: Dict[str, Any]) -> None:
    configured = _safe_str(result.get("configured_base_url", ""), "")
    reachable = _safe_str(result.get("reachable_base_url", ""), "")
    attempts = result.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []

    print(f"[ollama_discover] Configured base URL: {configured}")
    print("[ollama_discover] Probe results:")
    for attempt in attempts:
        base_url = _safe_str(attempt.get("base_url", ""), "")
        ok = bool(attempt.get("ok", False))
        if ok:
            status_code = attempt.get("status_code")
            model_count = attempt.get("model_count")
            print(f"  - OK   {base_url} (status={status_code}, models={model_count})")
        else:
            error = _safe_str(attempt.get("error", ""), "")
            print(f"  - FAIL {base_url} ({error})")

    if reachable:
        host_name = _safe_str(urlsplit(reachable).hostname, "")
        quoted_url = shlex.quote(reachable)
        print("[ollama_discover] Reachable endpoint found.")
        print(f"[ollama_discover] export OLLAMA_BASE_URL={quoted_url}")
        if host_name:
            print(
                "[ollama_discover] Optional: export OLLAMA_HOST_GATEWAY="
                f"{shlex.quote(host_name)}"
            )
    else:
        print("[ollama_discover] No reachable endpoint found from candidate list.")
        print(
            "[ollama_discover] Try setting OLLAMA_BASE_URL manually to a host-reachable address, "
            "then rerun this script."
        )


async def run(args: argparse.Namespace) -> int:
    if args.connect_timeout <= 0 or args.total_timeout <= 0:
        print("[ollama_discover] Timeout values must be > 0.")
        return 2
    if args.connect_timeout > args.total_timeout:
        print("[ollama_discover] connect-timeout must be <= total-timeout.")
        return 2

    result = await discover_reachable_endpoint(
        configured_base_url=args.base_url,
        connect_timeout=args.connect_timeout,
        total_timeout=args.total_timeout,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_text_summary(result)

    return 0 if result.get("reachable_base_url") else 1


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    raise SystemExit(asyncio.run(run(cli_args)))
