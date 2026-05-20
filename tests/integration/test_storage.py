import pytest


@pytest.mark.skip(reason="requires running Postgres service")
class TestStorageIntegration:
    async def test_migration_creates_tables(self):
        pass

    async def test_message_upsert_idempotent(self):
        pass

    async def test_edit_history_append(self):
        pass

    async def test_soft_delete(self):
        pass

    async def test_thread_reconstruction(self):
        pass
