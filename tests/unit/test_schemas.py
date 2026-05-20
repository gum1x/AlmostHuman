from datetime import datetime, timezone

from core.constants import EventType, MessageType
from core.schemas import (
    CanonicalMessage,
    ForwardMetadata,
    MediaMetadata,
    RawTelegramEvent,
    SenderInfo,
)


class TestRawTelegramEvent:
    def test_basic_creation(self, sample_raw_event_data):
        event = RawTelegramEvent(**sample_raw_event_data)
        assert event.message_id == 123
        assert event.chat_id == -1001234567890
        assert event.event_type == EventType.NEW_MESSAGE

    def test_roundtrip_json(self, sample_raw_event_data):
        event = RawTelegramEvent(**sample_raw_event_data)
        json_data = event.model_dump(mode="json")
        restored = RawTelegramEvent(**json_data)
        assert restored.message_id == event.message_id
        assert restored.chat_id == event.chat_id
        assert restored.text == event.text

    def test_none_text(self, sample_raw_event_data):
        sample_raw_event_data["text"] = None
        event = RawTelegramEvent(**sample_raw_event_data)
        assert event.text is None

    def test_none_sender(self, sample_raw_event_data):
        sample_raw_event_data["sender_id"] = None
        sample_raw_event_data["sender_info"] = None
        event = RawTelegramEvent(**sample_raw_event_data)
        assert event.sender_id is None

    def test_with_forward(self, sample_raw_event_data):
        sample_raw_event_data["forward"] = {
            "from_id": 111,
            "from_name": "Forwarder",
            "from_chat_id": None,
            "from_message_id": 42,
            "date": "2026-05-19T10:00:00+00:00",
        }
        event = RawTelegramEvent(**sample_raw_event_data)
        assert event.forward.from_id == 111
        assert event.forward.from_name == "Forwarder"

    def test_with_media(self, sample_raw_event_data):
        sample_raw_event_data["media"] = {
            "media_type": "photo",
            "file_id": "abc123",
            "file_size": 102400,
            "width": 1920,
            "height": 1080,
        }
        event = RawTelegramEvent(**sample_raw_event_data)
        assert event.media.media_type == "photo"
        assert event.media.file_size == 102400

    def test_delete_event(self):
        event = RawTelegramEvent(
            event_type=EventType.DELETE,
            message_id=1,
            chat_id=-100999,
            timestamp=datetime.now(timezone.utc),
            deleted_message_ids=[10, 11, 12],
        )
        assert event.event_type == EventType.DELETE
        assert len(event.deleted_message_ids) == 3


class TestCanonicalMessage:
    def test_basic_creation(self):
        msg = CanonicalMessage(
            message_id=1,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime.now(timezone.utc),
            message_type=MessageType.TEXT,
            text_raw="hello",
            text_cleaned="hello",
        )
        assert msg.message_type == MessageType.TEXT
        assert msg.mention_list == []

    def test_defaults(self):
        msg = CanonicalMessage(
            message_id=1,
            chat_id=-100999,
            timestamp=datetime.now(timezone.utc),
        )
        assert msg.message_type == MessageType.TEXT
        assert msg.entity_list == []
        assert msg.raw_event == {}
