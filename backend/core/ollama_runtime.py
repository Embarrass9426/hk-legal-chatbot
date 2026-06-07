import asyncio
import ipaddress
import json
import os
import socket
from urllib.parse import urlsplit
from typing import Any, AsyncGenerator, Dict, List

import httpx


_ollama_autostart_attempted = False


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _format_exception(exc: Exception) -> str:
    base_message = _safe_str(exc, "").strip()
    if not base_message:
        base_message = repr(exc)
    else:
        base_message = f"{exc.__class__.__name__}: {base_message}"

    nested = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if nested:
        nested_message = _safe_str(nested, "").strip() or repr(nested)
        return f"{base_message}; cause={nested.__class__.__name__}: {nested_message}"

    return base_message


def _parse_scheme_port(base_url: str) -> tuple[str, int]:
    split = urlsplit(base_url)
    scheme = split.scheme or "http"
    port = split.port or 11434
    return scheme, port


def _build_url(scheme: str, host: str, port: int) -> str:
    normalized_host = host
    try:
        parsed_ip = ipaddress.ip_address(host)
        if parsed_ip.version == 6:
            normalized_host = f"[{host}]"
    except ValueError:
        pass
    return f"{scheme}://{normalized_host}:{port}"


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _normalize_host_candidate(value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""

    if "//" in candidate:
        split = urlsplit(candidate)
        candidate = _safe_str(split.hostname, "").strip()

    if not candidate:
        return ""

    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1].strip()

    if candidate.count(":") > 1:
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            return ""

    if ":" in candidate:
        host_part, _, port_part = candidate.rpartition(":")
        if host_part and port_part.isdigit():
            candidate = host_part.strip()

    if not candidate:
        return ""

    if any(ch.isspace() for ch in candidate):
        return ""

    return candidate


def _get_resolv_nameserver_ip() -> str:
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as file:
            for line in file:
                candidate = line.strip()
                if not candidate.startswith("nameserver "):
                    continue
                ip = candidate.split()[1].strip()
                if ip:
                    return ip
    except Exception:
        return ""
    return ""


def _resolve_host_to_ip(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except Exception:
        return ""


def _is_valid_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _is_wsl_runtime() -> bool:
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as file:
            return "microsoft" in file.read().lower()
    except Exception:
        return False


def _is_loopback_or_unspecified_ip(ip: str) -> bool:
    try:
        parsed = ipaddress.ip_address(ip)
        return parsed.is_loopback or parsed.is_unspecified
    except ValueError:
        return False


def _get_wsl_default_gateway_ip() -> str:
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as file:
            next(file, None)
            for line in file:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                destination = parts[1]
                gateway_hex = parts[2]
                if destination != "00000000" or len(gateway_hex) != 8:
                    continue
                octets = [str(int(gateway_hex[i : i + 2], 16)) for i in range(0, 8, 2)]
                octets.reverse()
                return ".".join(octets)
    except Exception:
        return ""
    return ""


def candidate_ollama_base_urls(base_url: str) -> List[str]:
    cleaned = (base_url or "").strip().rstrip("/")
    candidates: List[str] = []

    if cleaned:
        candidates.append(cleaned)

    split = urlsplit(cleaned)
    configured_host = _safe_str(split.hostname, "").strip().lower()
    scheme, port = _parse_scheme_port(cleaned)

    if _is_loopback_host(configured_host):
        candidates.extend(
            [
                _build_url(scheme, "localhost", port),
                _build_url(scheme, "127.0.0.1", port),
            ]
        )

        host_gateway = _normalize_host_candidate(os.getenv("OLLAMA_HOST_GATEWAY", ""))
        if host_gateway:
            candidates.append(_build_url(scheme, host_gateway, port))

        if _is_wsl_runtime():
            gateway_ip = _get_wsl_default_gateway_ip()
            if gateway_ip and not _is_loopback_or_unspecified_ip(gateway_ip):
                candidates.append(_build_url(scheme, gateway_ip, port))

            resolv_ip = _get_resolv_nameserver_ip()
            if resolv_ip and not _is_loopback_or_unspecified_ip(resolv_ip):
                candidates.append(_build_url(scheme, resolv_ip, port))

        docker_host = "host.docker.internal"
        candidates.append(_build_url(scheme, docker_host, port))
        resolved_docker_host = _resolve_host_to_ip(docker_host)
        if resolved_docker_host and _is_valid_ip(resolved_docker_host):
            candidates.append(_build_url(scheme, resolved_docker_host, port))

    deduped: List[str] = []
    seen = set()
    for url in candidates:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


async def _try_autostart_ollama() -> bool:
    global _ollama_autostart_attempted

    if _ollama_autostart_attempted:
        return False

    auto_start = os.getenv("OLLAMA_AUTO_START", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not auto_start:
        return False

    _ollama_autostart_attempted = True
    try:
        await asyncio.create_subprocess_exec(
            "ollama",
            "serve",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(1.0)
        return True
    except Exception:
        return False


def _build_connect_error(
    scope: str, configured: str, errors: List[str]
) -> RuntimeError:
    host_name = _safe_str(socket.gethostname(), "unknown-host")
    error_report = (
        " | ".join(errors) if errors else "no endpoint attempts were recorded"
    )
    return RuntimeError(
        f"Unable to connect to Ollama {scope}. "
        f"Host={host_name}. Checked OLLAMA_BASE_URL candidates derived from configured endpoint. "
        f"Attempt errors: {error_report}. "
        "Hints: ensure Ollama is running and reachable from this runtime; if Ollama runs on the host OS, "
        "set OLLAMA_BASE_URL or OLLAMA_HOST_GATEWAY to the host-reachable address."
    )


async def verify_ollama_ready(configured_base_url: str) -> None:
    timeout = httpx.Timeout(8.0, connect=3.0)
    errors: List[str] = []

    for round_index in range(2):
        for base_url in candidate_ollama_base_urls(configured_base_url):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(f"{base_url}/api/tags")
                    response.raise_for_status()
                    return
            except Exception as exc:
                errors.append(f"{base_url}: {_format_exception(exc)}")

        if round_index == 0:
            started = await _try_autostart_ollama()
            if started:
                continue

    raise _build_connect_error("/api/tags", configured_base_url, errors)


async def post_ollama_chat_with_fallback(
    payload: Dict[str, Any], configured_base_url: str
) -> Dict[str, Any]:
    timeout = httpx.Timeout(120.0, connect=5.0)
    errors: List[str] = []

    for round_index in range(2):
        for base_url in candidate_ollama_base_urls(configured_base_url):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(f"{base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    try:
                        return response.json()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Received non-JSON /api/chat response from {base_url}"
                        ) from exc
            except Exception as exc:
                errors.append(f"{base_url}: {_format_exception(exc)}")

        if round_index == 0:
            started = await _try_autostart_ollama()
            if started:
                continue

    raise _build_connect_error("/api/chat", configured_base_url, errors)


async def stream_ollama_chat_with_fallback(
    payload: Dict[str, Any], configured_base_url: str
) -> AsyncGenerator[str, None]:
    timeout = httpx.Timeout(300.0 if payload.get("think") else 120.0, connect=5.0)
    errors: List[str] = []

    for round_index in range(2):
        for base_url in candidate_ollama_base_urls(configured_base_url):
            emitted_any_chunk = False
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", f"{base_url}/api/chat", json=payload
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            chunk_line = line.strip()
                            if not chunk_line:
                                continue
                            data = json.loads(chunk_line)
                            message_obj = data.get("message", {})
                            content_text = _safe_str(message_obj.get("content", ""), "")
                            thinking_text = _safe_str(message_obj.get("thinking", ""), "")
                            if thinking_text:
                                print(f"[Ollama] Got thinking chunk: {thinking_text[:50]}...")
                            if thinking_text and not content_text:
                                chunk_text = f"<thinking>{thinking_text}</thinking>"
                            elif thinking_text and content_text:
                                chunk_text = f"<thinking>{thinking_text}</thinking>{content_text}"
                            else:
                                chunk_text = content_text
                            if chunk_text:
                                emitted_any_chunk = True
                                yield chunk_text
                            if bool(data.get("done", False)):
                                return
            except Exception as exc:
                if emitted_any_chunk:
                    raise RuntimeError(
                        "Ollama streaming response interrupted after partial output; "
                        "fallback disabled to avoid duplicate generations."
                    ) from exc
                errors.append(f"{base_url}: {_format_exception(exc)}")

        if round_index == 0:
            started = await _try_autostart_ollama()
            if started:
                continue

    raise _build_connect_error("/api/chat (stream)", configured_base_url, errors)
