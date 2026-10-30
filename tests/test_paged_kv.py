"""Page pool bookkeeping (no MLX needed)."""

import random

import pytest
from instinct.paged_kv import KVGroup, PagePool


def test_lowest_free_page_first_keeps_the_arena_compact():
    pool = PagePool(8)
    pages = [pool.allocate(reserved=False) for _ in range(3)]
    assert pages == [0, 1, 2] and pool.high_water == 3
    pool.release([1])
    assert pool.allocate(reserved=False) == 1
    assert pool.high_water == 3


def test_reservations_hold_pages_for_their_owner():
    pool = PagePool(4)
    assert pool.reserve(3)
    assert pool.available == 1 and not pool.reserve(2)
    pool.allocate(reserved=True)
    assert pool.reserved == 2 and pool.free == 3 and pool.available == 1
    pool.allocate(reserved=False)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.allocate(reserved=False)
    pool.unreserve(2)
    assert pool.available == 2
    with pytest.raises(RuntimeError):
        pool.unreserve(1)


def test_shared_pages_return_only_after_the_last_reference():
    pool = PagePool(2)
    page = pool.allocate(reserved=False)
    pool.retain([page, -1])
    assert pool.shared(page)
    pool.release([page])
    assert pool.free == 1 and not pool.shared(page)
    pool.release([page, -1])
    assert pool.free == 2
    with pytest.raises(RuntimeError):
        pool.release([page])
    with pytest.raises(RuntimeError):
        pool.retain([page])


@pytest.mark.parametrize("seed", range(5))
def test_random_traffic_never_loses_or_duplicates_pages(seed):
    rng = random.Random(seed)
    pool = PagePool(32)
    held: list[int] = []
    for _ in range(500):
        action = rng.random()
        if action < 0.5 and pool.available:
            held.append(pool.allocate(reserved=False))
        elif action < 0.7 and held:
            page = rng.choice(held)
            pool.retain([page])
            held.append(page)
        elif held:
            pool.release([held.pop(rng.randrange(len(held)))])
        assert pool.free + len(set(held)) == pool.capacity
    pool.release(held)
    assert pool.free == pool.capacity


def test_group_sizes():
    group = KVGroup("full", (3, 7), kv_heads=2, head_dim=512)
    assert group.bytes_per_token(2) == 2 * 2 * 2 * 512 * 2
