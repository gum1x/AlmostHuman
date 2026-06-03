# Test UI

## VPS (already deployed)

Start:
```
ssh vps 'bash ~/telegram-ci/start-test-ui.sh'
```

Open: http://100.89.201.13:7777

Stop:
```
ssh vps 'pkill -f test-ui/server.py'
```

Logs:
```
ssh vps 'tail -50 ~/telegram-ci/test-ui.log'
```

Redeploy after code changes:
```
rsync -avz test-ui/ vps:~/telegram-ci/test-ui/
ssh vps 'bash ~/telegram-ci/start-test-ui.sh'
```

## Local

```
bash test-ui/run.sh
```
Opens on http://localhost:7777

Needs python3.11+ (`/opt/homebrew/bin/python3.11`). Override with `PYTHON=/path/to/python bash test-ui/run.sh`.

## Usage

1. Drag JSON chat files into the sidebar
2. Click a chat to view messages
3. "Run Pipeline" runs the full workflow (enrichment > gate > context > perception > decision > validation)
4. Type messages in the input to chat — bot responds through the real pipeline
5. Pipeline steps show in the right panel, click to expand details

## JSON format

Array of messages, or `{ "messages": [...] }`:

```json
[
  {
    "message_id": 1,
    "chat_id": -1001234567890,
    "sender_id": 12345,
    "text": "yo",
    "reply_to_message_id": null,
    "timestamp": "2026-06-01T12:00:00Z"
  }
]
```

Only `message_id` and `text` are required. The rest have defaults.

## Notes

- With `XAI_API_KEY` set in `.env`, uses real Grok for perception + decision
- Without it, uses FakeAiClient (always returns silent) — good for testing pipeline mechanics
- Server binds 0.0.0.0:7777 so any device on the network can access it
