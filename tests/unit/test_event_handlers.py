from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from core.constants import EventType


class TestNewMessageHandler:
    def test_extracts_basic_fields(self, sample_raw_event_data):
        from core.schemas import RawTelegramEvent

        event = RawTelegramEvent(**sample_raw_event_data)
        assert event.event_type == EventType.NEW_MESSAGE
        assert event.message_id == 123
        assert event.sender_id == 987654321
        assert event.text == "Hello @world this is a test"

    def test_extracts_reply_to(self, sample_raw_event_data):
        from core.schemas import RawTelegramEvent

        sample_raw_event_data["reply_to_message_id"] = 99
        event = RawTelegramEvent(**sample_raw_event_data)
        assert event.reply_to_message_id == 99


class TestMediaExtraction:
    def test_photo_media(self):
        from ingestion.event_handlers import _extract_media
        from telethon import types

        photo = MagicMock()
        photo.id = 12345
        photo.size = 1024
        size_obj = MagicMock()
        size_obj.w = 800
        size_obj.h = 600
        photo.sizes = [size_obj]

        media = types.MessageMediaPhoto(photo=photo, ttl_seconds=None)
        result = _extract_media(media)

        assert result is not None
        assert result.media_type == "photo"
        assert result.width == 800
        assert result.height == 600

    def test_none_media(self):
        from ingestion.event_handlers import _extract_media

        result = _extract_media(None)
        assert result is None


class TestForwardExtraction:
    def test_user_forward(self):
        from ingestion.event_handlers import _extract_forward
        from telethon import types

        fwd = MagicMock()
        fwd.from_id = types.PeerUser(user_id=111)
        fwd.from_name = "TestUser"
        fwd.channel_post = None
        fwd.date = datetime(2026, 1, 1, tzinfo=timezone.utc)

        result = _extract_forward(fwd)
        assert result is not None
        assert result.from_id == 111
        assert result.from_name == "TestUser"

    def test_none_forward(self):
        from ingestion.event_handlers import _extract_forward

        result = _extract_forward(None)
        assert result is None
