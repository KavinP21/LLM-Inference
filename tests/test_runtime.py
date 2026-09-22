import pytest

from forge_llm.runtime import IterationScheduler, KVBlockPool, SequenceState


def test_block_boundaries_and_fragmentation() -> None:
    cache = KVBlockPool(capacity_bytes=4 * 512, bytes_per_token=32)
    assert cache.reserve(1, 33)
    cache.ensure_tokens(1, 15)
    assert cache.stats()["allocated_blocks"] == 1
    assert cache.stats()["internal_fragmentation_tokens"] == 1
    cache.ensure_tokens(1, 16)
    assert cache.stats()["allocated_blocks"] == 1
    cache.ensure_tokens(1, 17)
    assert cache.stats()["allocated_blocks"] == 2
    assert cache.stats()["internal_fragmentation_tokens"] == 15
    cache.release(1)
    assert cache.stats()["allocated_blocks"] == 0


def test_continuous_batch_order_and_reclamation() -> None:
    cache = KVBlockPool(capacity_bytes=8 * 512, bytes_per_token=32)
    scheduler = IterationScheduler(3, 64, 100, cache)
    first = scheduler.submit([1, 2], 4, [])
    second = scheduler.submit([3, 4], 4, [])
    third = scheduler.submit([5, 6], 4, [])

    schedule = scheduler.next()
    assert schedule.prefill == first and schedule.decode == ()
    scheduler.finish_prefill(first)
    scheduler.append_token(first, 10)

    schedule = scheduler.next()
    assert schedule.prefill == second and schedule.decode == (first,)
    scheduler.finish_prefill(second)
    scheduler.append_token(second, 11)

    schedule = scheduler.next()
    assert schedule.prefill == third and schedule.decode == (first, second)
    scheduler.cancel(first)
    scheduler.cancel(second)
    scheduler.cancel(third)
    assert cache.stats()["allocated_blocks"] == 0
    assert cache.stats()["reserved_blocks"] == 0


def test_reservations_prevent_mid_generation_overcommit() -> None:
    cache = KVBlockPool(capacity_bytes=2 * 512, bytes_per_token=32)
    scheduler = IterationScheduler(3, 32, 100, cache)
    first = scheduler.submit([1], 15, [])
    second = scheduler.submit([2], 15, [])
    with pytest.raises(ValueError, match="insufficient KV capacity"):
        scheduler.submit([3], 1, [])
    assert cache.stats()["reserved_blocks"] == 2
    scheduler.cancel(first)
    replacement = scheduler.submit([3], 1, [])
    assert scheduler.request(replacement).state is SequenceState.WAITING
    scheduler.cancel(second)
    scheduler.cancel(replacement)


def test_eos_completes_and_releases_capacity() -> None:
    cache = KVBlockPool(capacity_bytes=512, bytes_per_token=32)
    scheduler = IterationScheduler(1, 16, 100, cache)
    request_id = scheduler.submit([1, 2], 3, [9])
    assert scheduler.next().prefill == request_id
    scheduler.finish_prefill(request_id)
    assert scheduler.append_token(request_id, 9)
    assert scheduler.request(request_id).state is SequenceState.COMPLETED
    assert cache.stats()["allocated_blocks"] == 0
