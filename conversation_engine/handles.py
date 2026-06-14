from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


class HandleMap:
    """Sender_id -> @handle / display name lookup.

    Built from plain sender dicts ({sender_id, username, first_name, ...}).
    Pure and offline: no DB, no network. Used to turn the anonymous
    ``user_<id>`` rendering into a real ``@username`` mention when one exists.
    """

    def __init__(self, senders: Iterable[Mapping[str, Any]]):
        self._username: dict[int, str | None] = {}
        self._first_name: dict[int, str | None] = {}
        for sender in senders:
            sid = sender.get("sender_id")
            if sid is None:
                continue
            sid = int(sid)
            self._username[sid] = sender.get("username") or None
            self._first_name[sid] = sender.get("first_name") or None

    @classmethod
    def from_jsonl(cls, path: str) -> "HandleMap":
        """Build from a senders.jsonl file (one JSON sender per line)."""
        senders: list[Mapping[str, Any]] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                senders.append(json.loads(line))
        return cls(senders)

    def handle_for(self, sender_id: int) -> str | None:
        """Return ``'@username'`` for the sender, or None when no username is known."""
        username = self._username.get(int(sender_id))
        if not username:
            return None
        return f"@{username}"

    def display_for(self, sender_id: int) -> str:
        """Best display string: username, else first_name, else ``user_<id>``."""
        sid = int(sender_id)
        username = self._username.get(sid)
        if username:
            return username
        first_name = self._first_name.get(sid)
        if first_name:
            return first_name
        return f"user_{sid}"
