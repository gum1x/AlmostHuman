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

from conversation_engine.config import load_engine_config
from runner import run_pipeline, PipelineResult, StepResult

app = FastAPI(title="Conversation Engine Test UI")

# Serve static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory store of uploaded chats (session-scoped)
_chats: dict[str, list[dict[str, Any]]] = {}
_config = None


def _get_config():
    global _config
    if _config is None:
        _config = load_engine_config(PROJECT_ROOT / "config.toml")
    return _config


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


@app.get("/api/chat/{name}")
async def get_chat(name: str):
    if name not in _chats:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    return {"name": name, "messages": _chats[name]}


@app.delete("/api/chat/{name}")
async def delete_chat(name: str):
    _chats.pop(name, None)
    return {"status": "ok"}


@app.post("/api/run/{name}")
async def run_chat(name: str, payload: dict[str, Any] | None = None):
    """Run the full pipeline on a chat. Optionally pass {"target_message_id": N} to target a specific message."""
    if name not in _chats:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    config = _get_config()
    target_id = (payload or {}).get("target_message_id")
    result = await run_pipeline(_chats[name], config=config, target_message_id=target_id)
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

    # Run pipeline
    config = _get_config()
    result = await run_pipeline(msgs, config=config)

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

        # Read optional initial message for target_message_id
        target_id = None
        try:
            init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
            target_id = init_msg.get("target_message_id")
        except (asyncio.TimeoutError, Exception):
            pass

        config = _get_config()
        result = await run_pipeline(_chats[name], config=config, target_message_id=target_id)

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
