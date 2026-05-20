from core.utils import clean_text, extract_mentions


class TestCleanText:
    def test_removes_null_bytes(self):
        assert clean_text("hello\x00world") == "helloworld"

    def test_removes_zero_width(self):
        assert clean_text("hello\u200bworld") == "helloworld"

    def test_normalizes_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_normalizes_newlines(self):
        result = clean_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strips(self):
        assert clean_text("  hello  ") == "hello"

    def test_none(self):
        assert clean_text(None) is None


class TestExtractMentions:
    def test_single_mention(self):
        assert extract_mentions("hey @alice") == ["alice"]

    def test_multiple_mentions(self):
        result = extract_mentions("@alice and @bob_test")
        assert "alice" in result
        assert "bob_test" in result

    def test_no_mentions(self):
        assert extract_mentions("no mentions here") == []

    def test_none(self):
        assert extract_mentions(None) == []

    def test_empty(self):
        assert extract_mentions("") == []
