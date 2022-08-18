"""Tests for tueopt/__init__.py."""

import time

import pytest

import tueopt

NAMES = ["world", "github"]
IDS = NAMES


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello(name):
    """Test hello function."""
    tueopt.hello(name)


@pytest.mark.expensive
@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello_expensive(name):
    """Expensive test of hello. Will only be run on master/main and development."""
    time.sleep(1)
    tueopt.hello(name)
