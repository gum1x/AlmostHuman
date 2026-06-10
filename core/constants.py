from enum import StrEnum


class MessageType(StrEnum):
    TEXT = "text"
    PHOTO = "photo"
    VIDEO = "video"
    DOCUMENT = "document"
    STICKER = "sticker"
    VOICE = "voice"
    VIDEO_NOTE = "video_note"
    ANIMATION = "animation"
    CONTACT = "contact"
    LOCATION = "location"
    POLL = "poll"
    OTHER = "other"


class EventType(StrEnum):
    NEW_MESSAGE = "new_message"
    EDIT = "edit"
    DELETE = "delete"
    CHAT_ACTION = "chat_action"
    REACTION_UPDATE = "reaction_update"
