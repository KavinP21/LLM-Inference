import math

import pytest
from forge_llm.runtime import IterationScheduler, KVBlockPool, SequenceState


def test_block_boundaries_and_fragmentation() -> None:
    cache = KVBlockPool(capacity_bytes=4 * 512, bytes_per_token=32)
    assert cache.reserve(1, 33)
    cache.ensure_tokens(1, 15)
    first_block = cache.block_table(1)
    assert len(first_block) == 1
    assert cache.stats()["allocated_blocks"] == 1
    assert cache.stats()["internal_fragmentation_tokens"] == 1
    cache.ensure_tokens(1, 16)
    assert cache.stats()["allocated_blocks"] == 1
    cache.ensure_tokens(1, 17)
    assert cache.block_table(1)[0] == first_block[0]
    assert len(set(cache.block_table(1))) == 2
    assert cache.stats()["allocated_blocks"] == 2
    assert cache.stats()["internal_fragmentation_tokens"] == 15
    cache.release(1)
    assert cache.stats()["allocated_blocks"] == 0


def test_physical_block_ids_are_reclaimed_and_reused() -> None:
    cache = KVBlockPool(capacity_bytes=2 * 512, bytes_per_token=32, layout="paged")
    assert cache.reserve(1, 17)
    cache.ensure_tokens(1, 17)
    released = cache.release(1)
    assert len(set(released)) == 2

    assert cache.reserve(2, 17)
    cache.ensure_tokens(2, 17)
    assert set(cache.block_table(2)) == set(released)
    assert cache.stats()["layout"] == "paged"


@pytest.mark.parametrize("tokens", [15, 16, 17, 31, 32, 33])
def test_allocator_block_boundary_matrix(tokens: int) -> None:
    cache = KVBlockPool(capacity_bytes=4 * 512, bytes_per_token=32, layout="paged")
    assert cache.reserve(1, 64)
    cache.ensure_tokens(1, tokens)
    assert len(cache.block_table(1)) == math.ceil(tokens / 16)
    assert cache.stats()["internal_fragmentation_tokens"] == (
        math.ceil(tokens / 16) * 16 - tokens
    )


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


def test_active_prefill_remains_scheduled_until_explicitly_finished() -> None:
    cache = KVBlockPool(capacity_bytes=4 * 512, bytes_per_token=32)
    scheduler = IterationScheduler(2, 64, 100, cache)
    first = scheduler.submit([1, 2, 3], 2, [])
    second = scheduler.submit([4, 5], 2, [])

    assert scheduler.next().prefill == first
    assert scheduler.next().prefill == first
    scheduler.finish_prefill(first)
    scheduler.append_token(first, 6)
    schedule = scheduler.next()
    assert schedule.prefill == second
    assert schedule.decode == (first,)


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
