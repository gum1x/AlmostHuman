"""
Test UI server — FastAPI backend that serves the chat testing interface
and runs the conversation engine pipeline on uploaded JSON chat data.

Usage:
    cd /path/to/Research.
    python -m test-ui.server          # OR
    python test-ui/server.py

Access from any device on your network at http://<YOUR_IP>:7777
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root AND test-ui dir are on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_UI_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(TEST_UI_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

from conversation_engine.config import load_engine_config
from runner import run_pipeline, PipelineResult, StepResult, TUNABLE_META, READONLY_SIGNALS

app = FastAPI(title="Conversation Engine Test UI")

# Serve static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory store of uploaded chats (session-scoped)
_chats: dict[str, list[dict[str, Any]]] = {}
# Previous bot memories (posture, responses, reasoning) for carry-over across turns
# and for re-runs with different overrides on the same chat snapshot.
_bot_mems: dict[str, list[dict[str, Any]]] = {}
def _get_config():
    # Re-read on every request (cheap: a TOML parse + dotenv) so the test-UI always
    # reflects the live .env — e.g. flipping CLOUD_BRAIN_ENABLED / TIMING_CLASSIFIER_*
    # takes effect without restarting the server. (Previously cached once, which left
    # the UI stuck on a stale config after the .env changed.)
    #
    # override=True makes the .env file authoritative even if the server process
    # inherited a stale value for the same var in its shell environment. This is safe
    # here because the test-UI runs on the host where .env is the single source of
    # truth (the containerized engine, by contrast, gets its vars from compose).
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    return load_engine_config(PROJECT_ROOT / "config.toml")


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---- Routes ----

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.post("/api/upload")
async def upload_chat(payload: dict[str, Any]):
    """Upload a JSON chat. Body: { "name": "chat_name", "messages": [...] }"""
    name = payload.get("name", "unnamed")
    messages = payload.get("messages", [])
    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)
    _chats[name] = messages
    return {"status": "ok", "name": name, "message_count": len(messages)}


@app.get("/api/chats")
async def list_chats():
    return {
        name: {
            "message_count": len(msgs),
            "preview": msgs[-1].get("text", msgs[-1].get("text_raw", ""))[:80] if msgs else "",
        }
        for name, msgs in _chats.items()
    }


@app.get("/api/tunables")
async def get_tunables():
    """Return the definition of what can be edited in the test UI (friendly labels, ranges, only changeable things)."""
    return {
        "readonly": list(READONLY_SIGNALS),
        "tunables": TUNABLE_META,
    }


@app.get("/api/chat/{name}")
async def get_chat(name: str):
    if name not in _chats:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    return {"name": name, "messages": _chats[name]}


@app.delete("/api/chat/{name}")
async def delete_chat(name: str):
    _chats.pop(name, None)
    return {"status": "ok"}


@app.post("/api/reset/{name}")
async def reset_chat(name: str, payload: dict[str, Any] | None = None):
    """Reset chat messages (and clear bot mems for clean 'from 0' sims).
    Optional body { "messages": [...] } to seed a base transcript (e.g. original user turns).
    Always clears _bot_mems[name] so subsequent sends start with fresh posture/activity state.
    """
    msgs = (payload or {}).get("messages")
    _chats[name] = list(msgs) if msgs is not None else []
    _bot_mems[name] = []
    return {"status": "ok", "name": name, "message_count": len(_chats[name])}


@app.post("/api/run/{name}")
async def run_chat(name: str, payload: dict[str, Any] | None = None):
    """Run the full pipeline on a chat. Pass {"overrides": {...}, "target_message_id": N} for live tuning."""
    if name not in _chats:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    config = _get_config()
    target_id = (payload or {}).get("target_message_id")
    overrides = (payload or {}).get("overrides") or {}
    prev = _bot_mems.get(name, [])
    result = await run_pipeline(
        _chats[name],
        config=config,
        target_message_id=target_id,
        previous_bot_memories=prev,
        overrides=overrides,
    )
    return _serialize_result(result)


@app.post("/api/send/{name}")
async def send_message(name: str, payload: dict[str, Any]):
    """Add a user message to the chat and run the pipeline."""
    if name not in _chats:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Append user message
    msgs = _chats[name]
    last_id = max((m.get("message_id", 0) for m in msgs), default=0)
    new_msg = {
        "message_id": last_id + 1,
        "chat_id": msgs[0].get("chat_id", -1001234567890) if msgs else -1001234567890,
        "sender_id": payload.get("sender_id", 999999),
        "text": text,
        "text_raw": text,
        "text_cleaned": text,
        "reply_to_message_id": payload.get("reply_to_message_id"),
        "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }
    msgs.append(new_msg)

    # Run pipeline (with current mem state + any overrides from payload)
    config = _get_config()
    overrides = payload.get("overrides") or {}
    prev = _bot_mems.get(name, [])
    result = await run_pipeline(msgs, config=config, previous_bot_memories=prev, overrides=overrides)

    # If bot responded, append bot message to chat
    bot_msg = None
    if result.response_text:
        bot_msg = {
            "message_id": last_id + 2,
            "chat_id": new_msg["chat_id"],
            "sender_id": 0,  # bot
            "text": result.response_text,
            "text_raw": result.response_text,
            "text_cleaned": result.response_text,
            "reply_to_message_id": result.decision.get("reply_to_message_id") if result.decision else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_bot": True,
        }
        msgs.append(bot_msg)

    # Accumulate bot memory entry for posture / recent activity carry-over on future runs/sends
    if result.decision:
        mem_entry = {
            "current_posture": result.decision.get("updated_engagement_posture"),
            "response_text": result.response_text,
            "reasoning": result.decision.get("reasoning"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reply_to_user_id": result.decision.get("reply_to_user_id"),
            "sent_message_id": None,  # not real send in test
        }
        _bot_mems.setdefault(name, []).append(mem_entry)

    return {
        "user_message": new_msg,
        "bot_message": bot_msg,
        "pipeline": _serialize_result(result),
    }


@app.websocket("/ws/run/{name}")
async def ws_run(websocket: WebSocket, name: str):
    """WebSocket endpoint that streams pipeline steps in real-time.
    Send {"target_message_id": N} as first message to target a specific message."""
    await websocket.accept()
    try:
        if name not in _chats:
            await websocket.send_json({"type": "error", "message": "Chat not found"})
            await websocket.close()
            return

        # Read optional initial message for target_message_id + overrides (so you can set dials before clicking Run and have them apply even over WS)
        target_id = None
        overrides = {}
        try:
            init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
            target_id = init_msg.get("target_message_id")
            overrides = init_msg.get("overrides") or {}
        except (asyncio.TimeoutError, Exception):
            pass

        config = _get_config()
        prev = _bot_mems.get(name, [])
        result = await run_pipeline(
            _chats[name],
            config=config,
            target_message_id=target_id,
            previous_bot_memories=prev,
            overrides=overrides,
        )

        # Stream each step
        for step in result.steps:
            await websocket.send_json({
                "type": "step",
                "step": _serialize_step(step),
            })
            await asyncio.sleep(0.05)  # tiny delay for UI animation

        # Final result
        await websocket.send_json({
            "type": "complete",
            "response_text": result.response_text,
            "decision": result.decision,
            "total_duration_ms": result.total_duration_ms,
        })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ---- Serialization helpers ----

def _serialize_step(step: StepResult) -> dict[str, Any]:
    return {
        "name": step.name,
        "duration_ms": step.duration_ms,
        "data": step.data,
        "error": step.error,
    }


def _serialize_result(result: PipelineResult) -> dict[str, Any]:
    return {
        "chat_id": result.chat_id,
        "steps": [_serialize_step(s) for s in result.steps],
        "response_text": result.response_text,
        "decision": result.decision,
        "total_duration_ms": result.total_duration_ms,
    }


# ---- Entry point ----

if __name__ == "__main__":
    import uvicorn

    ip = _get_local_ip()
    port = 7777
    print(f"\n  Test UI available at:")
    print(f"    Local:   http://localhost:{port}")
    print(f"    Network: http://{ip}:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
