import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False, help="Run slow tests (require live LLM)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="Skipped by default — run with --slow to include")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
