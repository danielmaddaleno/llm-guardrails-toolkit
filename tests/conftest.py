"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return {"key": "value", "count": 42}
