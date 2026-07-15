"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def text_with_pii():
    """A sentence containing an email, a phone number, and an SSN."""
    return "Reach John at john.doe@example.com or 555-123-4567, SSN 123-45-6789."
