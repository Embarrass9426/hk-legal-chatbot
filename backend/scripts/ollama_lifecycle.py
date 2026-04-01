import argparse
import asyncio
import os
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.core.ollama_runtime import (
    post_ollama_chat_with_fallback,
    verify_ollama_ready,
)


async def warm_model(model: str, base_url: str, keep_alive: str) -> None:
    await verify_ollama_ready(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 1,
            "num_ctx": 128000,
            "num_predict": 1,
        },
    }
    await post_ollama_chat_with_fallback(payload, base_url)
    print(f"[ollama_lifecycle] Warmed model '{model}' with keep_alive={keep_alive}")


async def unload_model(model: str, base_url: str) -> None:
    await verify_ollama_ready(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "unload"}],
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {
            "temperature": 1,
            "num_ctx": 128000,
            "num_predict": 1,
        },
    }
    await post_ollama_chat_with_fallback(payload, base_url)
    print(f"[ollama_lifecycle] Unload requested for model '{model}'")


async def show_status(base_url: str) -> None:
    await verify_ollama_ready(base_url)
    print(f"[ollama_lifecycle] Ollama reachable via OLLAMA_BASE_URL={base_url}")


async def run(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")

    if args.action == "status":
        await show_status(base_url)
        return

    if args.action == "start":
        await warm_model(args.model, base_url, args.keep_alive)
        return

    if args.action == "stop":
        await unload_model(args.model, base_url)
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage Ollama model lifecycle (warm, unload, status)."
    )
    parser.add_argument(
        "action",
        choices=["start", "stop", "status"],
        help="Lifecycle action",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:9b"),
        help="Ollama model name",
    )
    parser.add_argument(
        "--keep-alive",
        default="5m",
        help="Keep model loaded duration after request",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Preferred Ollama base URL",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    asyncio.run(run(cli_args))
