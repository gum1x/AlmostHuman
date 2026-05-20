import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_timestamp():
    return datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_raw_event_data(sample_timestamp):
    return {
        "event_type": "new_message",
        "message_id": 123,
        "chat_id": -1001234567890,
        "sender_id": 987654321,
        "timestamp": sample_timestamp.isoformat(),
        "text": "Hello @world this is a test",
        "reply_to_message_id": None,
        "forward": None,
        "media": None,
        "entities": [],
        "grouped_id": None,
        "sender_info": {
            "sender_id": 987654321,
            "username": "testuser",
            "first_name": "Test",
            "last_name": "User",
            "is_bot": False,
            "is_premium": False,
        },
        "deleted_message_ids": [],
        "raw": {"id": 123, "chat_id": -1001234567890},
    }
