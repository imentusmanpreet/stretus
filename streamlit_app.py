"""
Standalone Streamlit tester for the Stretus FastAPI chat workflow.

This file intentionally does not import or modify the backend application code.
It talks to the running FastAPI server over HTTP.
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://localhost:8000"
API_PREFIX = "/api/v1/strategy"
REQUEST_TIMEOUT_SECONDS = 30


def _normalise_backend_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    return text or DEFAULT_BACKEND_URL


def _api_url(path: str) -> str:
    base_url = _normalise_backend_url(st.session_state.get("backend_url"))
    return f"{base_url}{API_PREFIX}{path}"


def _root_url(path: str) -> str:
    base_url = _normalise_backend_url(st.session_state.get("backend_url"))
    return f"{base_url}{path}"


def _request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach backend: {exc}") from exc

    if response.status_code == 204:
        return {}

    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}

    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        message = detail or payload.get("message") if isinstance(payload, dict) else None
        raise RuntimeError(message or f"Backend returned HTTP {response.status_code}.")

    return payload if isinstance(payload, dict) else {"data": payload}


def health_check() -> dict[str, Any]:
    return _request_json("GET", _root_url("/health"))


def list_chats(limit: int = 50) -> list[dict[str, Any]]:
    payload = _request_json("GET", _api_url("/chats"), params={"limit": limit})
    return list(payload.get("chats") or [])


def create_chat(title: str | None = None) -> dict[str, Any]:
    body = {"title": title} if title else {}
    return _request_json("POST", _api_url("/chats"), json=body)


def delete_chat(session_id: str) -> None:
    _request_json("DELETE", _api_url(f"/chats/{session_id}"))


def get_messages(session_id: str) -> dict[str, Any]:
    return _request_json("GET", _api_url(f"/chats/{session_id}/messages"))


def send_message(session_id: str, content: str) -> dict[str, Any]:
    return _request_json(
        "POST",
        _api_url(f"/chats/{session_id}/messages"),
        json={"content": content},
    )


def _safe_rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()


def _latest_assistant_status(history: dict[str, Any]) -> str | None:
    for message in reversed(history.get("messages") or []):
        if message.get("role") == "assistant":
            return message.get("status")
    return None


def _count_assistant_messages(history: dict[str, Any] | None) -> int:
    return sum(
        1
        for message in (history or {}).get("messages") or []
        if message.get("role") == "assistant"
    )


def poll_until_ready(
    session_id: str,
    *,
    baseline_assistant_count: int = 0,
    max_seconds: int = 90,
    interval_seconds: float = 1.5,
) -> dict[str, Any]:
    """Poll the chat history until a NEW assistant reply (newer than `baseline_assistant_count`)
    has finished processing. Without the baseline check, polling would exit immediately
    on the first iteration because the previous turn's assistant message is already
    ``completed`` — that is why the user had to click Refresh to see the new reply.
    """
    started_at = time.time()
    status_box = st.empty()
    progress_bar = st.progress(0)
    latest_history: dict[str, Any] = {}

    while time.time() - started_at < max_seconds:
        latest_history = get_messages(session_id)
        new_assistant_count = _count_assistant_messages(latest_history)
        latest_status = _latest_assistant_status(latest_history)
        elapsed = time.time() - started_at
        progress_bar.progress(min(1.0, elapsed / max_seconds))
        if new_assistant_count <= baseline_assistant_count:
            status_box.caption("Waiting for assistant response... (processing)")
        else:
            status_box.caption(
                f"Waiting for assistant response... (status: {latest_status or 'pending'})"
            )
        if (
            new_assistant_count > baseline_assistant_count
            and latest_status in {"completed", "failed"}
        ):
            break
        time.sleep(interval_seconds)

    progress_bar.empty()
    status_box.empty()
    return latest_history


def _format_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _message_label(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "assistant")
    state = message.get("state")
    status = message.get("status")
    bits = [role]
    if state:
        bits.append(str(state))
    if status:
        bits.append(str(status))
    return " | ".join(bits)


def render_message(message: dict[str, Any]) -> None:
    role = "assistant" if message.get("role") == "assistant" else "user"
    avatar = "assistant" if role == "assistant" else "user"
    with st.chat_message(role, avatar=avatar):
        st.caption(_message_label(message))
        content = str(message.get("content") or "")
        if message.get("error_message"):
            st.error(message["error_message"])
        if content:
            st.markdown(content)
        else:
            st.info("No message content returned yet.")

        for key, label in (
            ("strategy_draft", "Strategy Draft"),
            ("strategy_json", "Strategy JSON"),
            ("backtest_result", "Backtest Result"),
            ("error", "Error Payload"),
        ):
            value = message.get(key)
            if value:
                with st.expander(label, expanded=False):
                    st.code(_format_json(value), language="json")


def load_current_history() -> dict[str, Any] | None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        return None
    history = get_messages(session_id)
    st.session_state["history"] = history
    return history


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Backend")
        st.text_input(
            "FastAPI base URL",
            key="backend_url",
            placeholder=DEFAULT_BACKEND_URL,
            help="Use only the root URL. The app adds /api/v1/strategy automatically.",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Check Health", use_container_width=True):
                try:
                    st.session_state["health"] = health_check()
                    st.session_state["last_error"] = None
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)
        with col_b:
            if st.button("Refresh", use_container_width=True):
                try:
                    st.session_state["chat_list"] = list_chats()
                    load_current_history()
                    st.session_state["last_error"] = None
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)

        if st.session_state.get("health"):
            st.success(_format_json(st.session_state["health"]))

        st.divider()
        st.header("Session")
        title = st.text_input("New chat title", value="Streamlit Test Strategy")
        if st.button("Create New Chat", type="primary", use_container_width=True):
            try:
                chat = create_chat(title=title)
                st.session_state["session_id"] = chat["session_id"]
                st.session_state["history"] = get_messages(chat["session_id"])
                st.session_state["chat_list"] = list_chats()
                st.session_state["last_error"] = None
                _safe_rerun()
            except Exception as exc:
                st.session_state["last_error"] = str(exc)

        try:
            chats = st.session_state.get("chat_list") or list_chats()
            st.session_state["chat_list"] = chats
        except Exception as exc:
            chats = []
            st.session_state["last_error"] = str(exc)

        if chats:
            options = [chat["session_id"] for chat in chats]
            labels = {
                chat["session_id"]: f"{chat.get('title') or 'Untitled'} ({chat.get('message_count', 0)} msgs)"
                for chat in chats
            }
            current = st.session_state.get("session_id")
            index = options.index(current) if current in options else 0
            selected = st.selectbox(
                "Load existing chat",
                options,
                index=index,
                format_func=lambda item: labels.get(item, item),
            )
            if selected and selected != st.session_state.get("session_id"):
                st.session_state["session_id"] = selected
                try:
                    load_current_history()
                    st.session_state["last_error"] = None
                    _safe_rerun()
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)

        if st.session_state.get("session_id"):
            st.text_input("Current session_id", value=st.session_state["session_id"], disabled=True)
            if st.button("Delete Current Chat", use_container_width=True):
                try:
                    delete_chat(st.session_state["session_id"])
                    st.session_state["session_id"] = None
                    st.session_state["history"] = None
                    st.session_state["chat_list"] = list_chats()
                    st.session_state["last_error"] = None
                    _safe_rerun()
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)

        st.divider()
        st.header("Polling")
        st.checkbox("Auto-poll after sending", key="auto_poll")
        st.slider("Poll timeout seconds", min_value=15, max_value=180, step=15, key="poll_timeout")


def render_chat_panel() -> None:
    session_id = st.session_state.get("session_id")
    history = st.session_state.get("history")

    if not session_id:
        st.info("Create or load a chat session from the sidebar.")
        return

    if history is None:
        try:
            history = load_current_history()
        except Exception as exc:
            st.error(str(exc))
            return

    st.subheader(history.get("title") or "Untitled")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Mode", history.get("current_mode") or "-")
    meta_cols[1].metric("Status", history.get("status") or "-")
    meta_cols[2].metric("Messages", history.get("message_count") or 0)
    meta_cols[3].metric("Session", str(session_id)[:8])

    for message in history.get("messages") or []:
        render_message(message)

    # If a previous turn timed out before the assistant reply finished, keep
    # polling on every rerun so the user never has to click Refresh manually.
    if st.session_state.get("awaiting_response"):
        baseline_count = int(st.session_state.get("awaiting_baseline_count") or 0)
        try:
            st.session_state["history"] = poll_until_ready(
                session_id,
                baseline_assistant_count=baseline_count,
                max_seconds=int(st.session_state.get("poll_timeout") or 90),
            )
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
        latest_status = _latest_assistant_status(st.session_state.get("history") or {})
        new_count = _count_assistant_messages(st.session_state.get("history"))
        if new_count > baseline_count and latest_status in {"completed", "failed"}:
            st.session_state["awaiting_response"] = False
            st.session_state["awaiting_baseline_count"] = None
            _safe_rerun()
        else:
            # Still processing on the backend — schedule another rerun.
            time.sleep(1.0)
            _safe_rerun()
            return

    prompt = st.chat_input("Type a strategy request, modification, rejection, or backtest approval")
    if prompt:
        try:
            baseline_count = _count_assistant_messages(history)
            send_message(session_id, prompt)
            if st.session_state.get("auto_poll", True):
                st.session_state["history"] = poll_until_ready(
                    session_id,
                    baseline_assistant_count=baseline_count,
                    max_seconds=int(st.session_state.get("poll_timeout") or 90),
                )
                latest_status = _latest_assistant_status(st.session_state["history"])
                new_count = _count_assistant_messages(st.session_state["history"])
                if not (new_count > baseline_count and latest_status in {"completed", "failed"}):
                    # Polling timed out before the assistant reply finished —
                    # let the next rerun continue waiting automatically.
                    st.session_state["awaiting_response"] = True
                    st.session_state["awaiting_baseline_count"] = baseline_count
            else:
                st.session_state["history"] = get_messages(session_id)
                st.session_state["awaiting_response"] = True
                st.session_state["awaiting_baseline_count"] = baseline_count
            st.session_state["last_error"] = None
            _safe_rerun()
        except Exception as exc:
            st.session_state["last_error"] = str(exc)


def render_quick_prompts() -> None:
    st.caption("Quick prompts")
    prompts = [
        "Create an intraday bullish INFY strategy on 15m. I am intermediate and want breakout with controlled risk.",
        "Yes, plan the signals.",
        "This strategy is not good. Pause and suggest pivot options.",
        "Change timeframe to 5m and keep everything else same.",
        "Run the backtest.",
    ]
    cols = st.columns(len(prompts))
    for index, prompt in enumerate(prompts):
        with cols[index]:
            if st.button(f"Prompt {index + 1}", use_container_width=True):
                st.session_state["pending_prompt"] = prompt

    if st.session_state.get("pending_prompt"):
        st.text_area("Selected prompt", value=st.session_state["pending_prompt"], height=80)
        if st.button("Send Selected Prompt", type="primary"):
            session_id = st.session_state.get("session_id")
            if not session_id:
                chat = create_chat(title="Streamlit Quick Prompt")
                session_id = chat["session_id"]
                st.session_state["session_id"] = session_id
            try:
                baseline_count = _count_assistant_messages(st.session_state.get("history"))
                send_message(session_id, st.session_state["pending_prompt"])
                st.session_state["pending_prompt"] = None
                if st.session_state.get("auto_poll", True):
                    st.session_state["history"] = poll_until_ready(
                        session_id,
                        baseline_assistant_count=baseline_count,
                        max_seconds=int(st.session_state.get("poll_timeout") or 90),
                    )
                    latest_status = _latest_assistant_status(st.session_state["history"])
                    new_count = _count_assistant_messages(st.session_state["history"])
                    if not (new_count > baseline_count and latest_status in {"completed", "failed"}):
                        st.session_state["awaiting_response"] = True
                        st.session_state["awaiting_baseline_count"] = baseline_count
                else:
                    st.session_state["history"] = get_messages(session_id)
                    st.session_state["awaiting_response"] = True
                    st.session_state["awaiting_baseline_count"] = baseline_count
                st.session_state["last_error"] = None
                _safe_rerun()
            except Exception as exc:
                st.session_state["last_error"] = str(exc)


def main() -> None:
    st.set_page_config(page_title="Stretus Chat Tester", page_icon=None, layout="wide")
    st.title("Stretus Chat Tester")
    st.caption("Standalone Streamlit UI for testing the current FastAPI chat and agent workflow.")

    st.session_state.setdefault("backend_url", DEFAULT_BACKEND_URL)
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("history", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("auto_poll", True)
    st.session_state.setdefault("poll_timeout", 90)
    st.session_state.setdefault("awaiting_response", False)
    st.session_state.setdefault("awaiting_baseline_count", None)

    render_sidebar()

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    render_quick_prompts()
    st.divider()
    render_chat_panel()


if __name__ == "__main__":
    main()
