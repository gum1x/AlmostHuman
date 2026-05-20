from datetime import datetime, timezone

from core.constants import EventType, MessageType
from core.schemas import ForwardMetadata, MediaMetadata, RawTelegramEvent, SenderInfo
from pipeline.workers import MessageWorker


class TestMessageWorkerTransform:
    def setup_method(self):
        self.worker = MessageWorker()

    def test_basic_text_message(self):
        event = RawTelegramEvent(
            event_type=EventType.NEW_MESSAGE,
            message_id=1,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            text="Hello world",
        )
        msg = self.worker._transform(event)
        assert msg.message_type == MessageType.TEXT
        assert msg.text_raw == "Hello world"
        assert msg.text_cleaned == "Hello world"

    def test_text_cleaning(self):
        event = RawTelegramEvent(
            event_type=EventType.NEW_MESSAGE,
            message_id=2,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            text="Hello\x00 \u200b  world\n\n\n\ntest",
        )
        msg = self.worker._transform(event)
        assert "\x00" not in msg.text_cleaned
        assert "\u200b" not in msg.text_cleaned
        assert "\n\n\n\n" not in msg.text_cleaned

    def test_mention_extraction(self):
        event = RawTelegramEvent(
            event_type=EventType.NEW_MESSAGE,
            message_id=3,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            text="Hey @alice and @bob_123 check this",
        )
        msg = self.worker._transform(event)
        assert "alice" in msg.mention_list
        assert "bob_123" in msg.mention_list

    def test_media_type_detection(self):
        event = RawTelegramEvent(
            event_type=EventType.NEW_MESSAGE,
            message_id=4,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            media=MediaMetadata(media_type="photo", file_id="abc"),
        )
        msg = self.worker._transform(event)
        assert msg.message_type == MessageType.PHOTO
        assert msg.media_type == "photo"

    def test_forward_metadata(self):
        event = RawTelegramEvent(
            event_type=EventType.NEW_MESSAGE,
            message_id=5,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            text="fwd msg",
            forward=ForwardMetadata(
                from_id=111,
                from_name="Someone",
                from_message_id=42,
                date=datetime(2026, 5, 19, tzinfo=timezone.utc),
            ),
        )
        msg = self.worker._transform(event)
        assert msg.forward_from_id == 111
        assert msg.forward_from_name == "Someone"
        assert msg.forward_from_message_id == 42

    def test_none_text(self):
        event = RawTelegramEvent(
            event_type=EventType.NEW_MESSAGE,
            message_id=6,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            text=None,
            media=MediaMetadata(media_type="sticker"),
        )
        msg = self.worker._transform(event)
        assert msg.text_raw is None
        assert msg.text_cleaned is None
        assert msg.message_type == MessageType.STICKER

    def test_idempotent_transform(self):
        event = RawTelegramEvent(
            event_type=EventType.NEW_MESSAGE,
            message_id=7,
            chat_id=-100999,
            sender_id=555,
            timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            text="same text",
        )
        msg1 = self.worker._transform(event)
        msg2 = self.worker._transform(event)
        assert msg1.model_dump() == msg2.model_dump()
