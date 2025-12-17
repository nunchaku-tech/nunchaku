import importlib.util
import os
from pathlib import Path

import pytest
import torch


def _debug(msg: str) -> None:
    # Opt-in debug output for local runs:
    #   NUNCHAKU_TEST_DEBUG=1 pytest -s -q tests/test_pin_memory_auto.py
    if os.environ.get("NUNCHAKU_TEST_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        print(msg)


def _load_utils_module():
    """
    Load `nunchaku/utils.py` directly by path to avoid importing `nunchaku/__init__.py`,
    which may pull in heavy optional deps (e.g. transformers) during test collection.
    """
    repo_root = Path(__file__).resolve().parents[1]
    utils_path = repo_root / "nunchaku" / "utils.py"
    spec = importlib.util.spec_from_file_location("nunchaku_utils", utils_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nku = _load_utils_module()


@pytest.fixture(autouse=True)
def _clear_pin_memory_auto_cache():
    # Ensure tests don't leak global cache state across runs.
    nku._PIN_MEMORY_AUTO_CACHE.clear()
    yield
    nku._PIN_MEMORY_AUTO_CACHE.clear()


def test_resolve_pin_memory_non_cuda_is_always_false(monkeypatch: pytest.MonkeyPatch):
    def _probe_should_not_run(_: torch.device) -> bool:  # pragma: no cover
        raise AssertionError("_auto_pin_memory_probe should not be called for non-CUDA devices")

    monkeypatch.setattr(nku, "_auto_pin_memory_probe", _probe_should_not_run)

    got_auto = nku.resolve_pin_memory("auto", "cpu")
    got_true = nku.resolve_pin_memory(True, "cpu")
    got_false = nku.resolve_pin_memory(False, "cpu")

    _debug(f"non-cuda: auto={got_auto} true={got_true} false={got_false}")

    assert got_auto is False
    assert got_true is False
    assert got_false is False


def test_resolve_pin_memory_bool_bypasses_auto_probe_on_cuda(monkeypatch: pytest.MonkeyPatch):
    def _probe_should_not_run(_: torch.device) -> bool:  # pragma: no cover
        raise AssertionError("_auto_pin_memory_probe should not be called when pin_memory is bool")

    monkeypatch.setattr(nku, "_auto_pin_memory_probe", _probe_should_not_run)

    got_true = nku.resolve_pin_memory(True, "cuda:0")
    got_false = nku.resolve_pin_memory(False, "cuda:0")
    _debug(f"cuda bool: True-> {got_true}, False-> {got_false}")

    assert got_true is True
    assert got_false is False


def test_resolve_pin_memory_auto_is_cached_per_device_index(monkeypatch: pytest.MonkeyPatch):
    calls: list[torch.device] = []

    def _probe(device: torch.device) -> bool:
        calls.append(device)
        return True

    monkeypatch.setattr(nku, "_auto_pin_memory_probe", _probe)

    # Same CUDA device index -> probe once, then cached.
    got_0_a = nku.resolve_pin_memory("auto", "cuda:0")
    got_0_b = nku.resolve_pin_memory("auto", "cuda:0")
    _debug(f"cuda:0 auto: first={got_0_a} second={got_0_b} probe_calls={len(calls)} cache={nku._PIN_MEMORY_AUTO_CACHE}")

    assert got_0_a is True
    assert got_0_b is True
    assert len(calls) == 1
    assert nku._PIN_MEMORY_AUTO_CACHE == {0: True}

    # Different device index -> probe again.
    got_1 = nku.resolve_pin_memory("auto", "cuda:1")
    _debug(f"cuda:1 auto: got={got_1} probe_calls={len(calls)} cache={nku._PIN_MEMORY_AUTO_CACHE}")

    assert got_1 is True
    assert len(calls) == 2
    assert nku._PIN_MEMORY_AUTO_CACHE == {0: True, 1: True}


def test_resolve_pin_memory_auto_cache_key_none_for_device_cuda(monkeypatch: pytest.MonkeyPatch):
    # torch.device("cuda") has device.index == None, and the cache uses that key.
    calls = 0

    def _probe(_: torch.device) -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(nku, "_auto_pin_memory_probe", _probe)

    got_a = nku.resolve_pin_memory("auto", "cuda")
    got_b = nku.resolve_pin_memory("auto", "cuda")
    _debug(f"cuda (index=None) auto: first={got_a} second={got_b} probe_calls={calls} cache={nku._PIN_MEMORY_AUTO_CACHE}")

    assert got_a is False
    assert got_b is False
    assert calls == 1
    assert nku._PIN_MEMORY_AUTO_CACHE == {None: False}


