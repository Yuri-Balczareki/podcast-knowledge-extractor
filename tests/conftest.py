import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "functional: slow tests that run real ML models")
