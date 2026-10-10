"""Long-lived workers may discard terminal histories without losing accounting."""

import pytest
from forge_llm.runtime import IterationScheduler, KVBlockPool


def test_forgetting_active_requests_is_rejected():
    pool = KVBlockPool(4096, 8)
    scheduler = IterationScheduler(2, 32, 64, pool)
    request = scheduler.submit([1, 2], 3, ())
    with pytest.raises(RuntimeError, match="active"):
        scheduler.forget(request)
    scheduler.cancel(request)
    before = scheduler.stats()
    scheduler.forget(request)
    assert scheduler.stats() == before
    assert pool.stats()["allocated_blocks"] == 0
    with pytest.raises(KeyError):
        scheduler.request(request)


def test_many_completed_requests_do_not_retain_token_history():
    pool = KVBlockPool(4096, 8)
    scheduler = IterationScheduler(1, 32, 64, pool)
    for _ in range(500):
        request = scheduler.submit([1, 2], 1, ())
        scheduler.next()
        scheduler.finish_prefill(request)
        assert scheduler.append_token(request, 3)
        scheduler.forget(request)
    assert scheduler.requests == {}
    assert scheduler.stats()["completed"] == 500
    assert pool.stats()["allocated_blocks"] == 0
