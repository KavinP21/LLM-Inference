from __future__ import annotations

import random

import numpy as np
import pytest

from forge_llm.prefix_cache import PrefixCache, SharedKVBlockPool
from forge_llm.runtime import IterationScheduler


class Store:
    def __init__(self):
        self.released = []

    def release(self, ids):
        self.released.extend(ids)

    def validate_prefix(self, blocks, tokens):
        pass


def pool(blocks=8):
    return SharedKVBlockPool(blocks * 16, 1, layout="paged")


def invariant(p):
    refs = {}
    for allocation in p.allocations.values():
        assert len(set(allocation.blocks)) == len(allocation.blocks)
        for b in allocation.blocks:
            refs[b] = refs.get(b, 0) + 1
    for b, counts in p._pins.items():
        refs[b] = refs.get(b, 0) + sum(counts.values())
    assert refs == p._refs
    assert len(p.free_blocks) == len(set(p.free_blocks))
    assert not set(p.free_blocks) & set(refs)
    assert set(p.free_blocks) | set(refs) == set(range(p.total_blocks))
    stats = p.stats()
    assert 0 <= p.committed_blocks() <= p.total_blocks
    assert stats["allocated_blocks"] + stats["free_blocks"] == p.total_blocks
    assert stats["unreserved_blocks"] >= 0
    assert stats["internal_fragmentation_tokens"] >= 0


def test_full_pages_share_without_duplicate_physical_accounting():
    p = pool()
    assert p.reserve(1, 48)
    p.ensure_tokens(1, 32)
    table = p.block_table(1)
    assert p.pin(table, 32)
    assert p.reserve_shared(2, 48, table, 32)
    p.ensure_tokens(2, 33)
    assert p.stats()["active_block_references"] == 5
    assert p.stats()["allocated_blocks"] == 3
    assert p.stats()["shared_blocks"] == 2
    assert p.stats()["reserved_blocks"] == 1
    assert p.release(1) == ()
    assert p.release(2) == (2,)
    assert set(p.unpin(table, 32)) == set(table)
    invariant(p)


def test_partial_page_branches_reserve_and_copy_before_writing():
    p = pool(5)
    assert p.reserve(1, 32)
    p.ensure_tokens(1, 15)
    table = p.block_table(1)
    assert p.pin(table, 15)
    p.release(1)
    assert p.reserve_shared(2, 16, table, 15)
    assert p.reserve_shared(3, 16, table, 15)
    assert p.stats()["cow_reserved_blocks"] == 2
    copies = []
    for request in (2, 3):
        p.ensure_tokens(request, 16)
        p.make_writable(request, 15, 1, lambda a, b: copies.append((a, b)))
    assert len(copies) == 2
    assert copies[0][0] == copies[1][0] == table[0]
    assert p.block_table(2) != p.block_table(3)
    assert p.stats()["cow_reserved_blocks"] == 0
    assert p.stats()["cow_copies"] == 2
    assert p.stats()["internal_fragmentation_tokens"] == 1
    invariant(p)
    p.release(2)
    p.release(3)
    p.unpin(table, 15)
    invariant(p)


def test_cow_spare_is_not_overcommitted_by_cache_publication():
    p = pool(1)
    assert p.reserve(1, 16)
    p.ensure_tokens(1, 15)
    assert not p.pin(p.block_table(1), 15)
    assert p.stats()["cached_blocks"] == 0
    p.ensure_tokens(1, 16)
    assert p.pin(p.block_table(1), 16)
    assert not p.reserve(2, 1)
    assert p.reserve_shared(2, 16, p.block_table(1), 16)
    invariant(p)


def test_unpinning_does_not_free_active_pages_and_clears_unused_cow_reserve():
    p = pool(3)
    p.reserve(1, 32)
    p.ensure_tokens(1, 15)
    table = p.block_table(1)
    assert p.pin(table, 15)
    assert p.unpin(table, 15) == ()
    assert p.stats()["cow_reserved_blocks"] == 0
    p.ensure_tokens(1, 16)
    p.make_writable(1, 15, 1, lambda *a: pytest.fail("exclusive page needs no copy"))
    invariant(p)


def test_clone_failure_is_transactional_and_never_recycles_owned_page():
    p = pool(4)
    p.reserve(1, 32)
    p.ensure_tokens(1, 15)
    table = p.block_table(1)
    assert p.pin(table, 15)
    p.ensure_tokens(1, 16)
    before = p.stats()

    def fail(*args):
        raise RuntimeError("injected clone failure")

    with pytest.raises(RuntimeError, match="injected"):
        p.make_writable(1, 15, 1, fail)
    assert p.block_table(1) == table
    assert p.stats() == before
    invariant(p)


def test_rewriting_full_shared_prefix_is_forbidden():
    p = pool(4)
    p.reserve(1, 32)
    p.ensure_tokens(1, 16)
    table = p.block_table(1)
    assert p.pin(table, 16)
    with pytest.raises(RuntimeError, match="immutable"):
        p.make_writable(1, 0, 1, lambda *a: None)
    with pytest.raises(ValueError):
        p.unpin((table[0], table[0]), 32)
    assert p.unpin(table, 16) == ()
    with pytest.raises(ValueError, match="released"):
        p.unpin(table, 16)
    invariant(p)


def test_cache_lookup_is_namespace_and_chunk_shape_safe():
    p = pool(12)
    p.reserve(1, 64)
    p.ensure_tokens(1, 64)
    cache = PrefixCache(
        p, Store(), max_blocks=8, max_entries=4, chunk_size=16, identity="model A"
    )
    row = np.arange(8, dtype=np.float32)
    prompt = list(range(33))
    for length in (16, 31, 33):
        assert cache.put(prompt[:length], "tenant A", p.block_table(1), row)
    assert cache.find(prompt, "tenant A").tokens == tuple(prompt)
    # A final 31-token prompt cannot be reused as a chunk prefix of a longer one.
    assert len(cache.find(prompt[:32], "tenant A").tokens) == 16
    assert cache.find(prompt, "tenant B") is None
    assert cache.find([999] + prompt[1:], "tenant A") is None
    assert not cache.put(prompt, "tenant A", p.block_table(1), row)
    assert not cache.find(prompt, "tenant A").logits.flags.writeable
    cache.clear("tenant B")
    assert cache.stats()["entries"] == 3
    cache.clear()
    assert p.stats()["cached_blocks"] == 0
    invariant(p)


def test_cache_budgets_count_unique_pages_and_bound_host_logits():
    p = pool(12)
    p.reserve(1, 64)
    p.ensure_tokens(1, 64)
    store = Store()
    cache = PrefixCache(
        p, store, max_blocks=2, max_entries=1, chunk_size=16, identity="model"
    )
    assert cache.put(list(range(16)), "a", p.block_table(1), np.zeros(8))
    assert cache.put(list(range(32)), "a", p.block_table(1), np.zeros(8))
    assert cache.stats()["entries"] == 1
    assert cache.stats()["cached_blocks"] == 2
    assert cache.stats()["host_logits_bytes"] == 32
    assert not cache.put(list(range(33)), "a", p.block_table(1), np.zeros(8))
    assert cache.stats()["capacity_skips"] == 1
    assert store.released == []  # Live producer still owns both pages.
    invariant(p)


def test_admission_evicts_cache_only_not_admitted_work():
    p = pool(3)
    p.reserve(1, 32)
    p.ensure_tokens(1, 32)
    store = Store()
    cache = PrefixCache(
        p, store, max_blocks=2, max_entries=2, chunk_size=16, identity="model"
    )
    cache.put(list(range(32)), "a", p.block_table(1), np.zeros(8))
    p.release(1)
    cache.make_admission_room(48, None)
    assert cache.stats()["entries"] == 0
    assert len(store.released) == 2
    assert p.reserve(2, 48)
    invariant(p)


def test_scheduler_can_admit_using_shared_prefix_credit():
    p = pool(4)
    s = IterationScheduler(2, 48, 100, p)
    first = s.submit(list(range(32)), 8, [])
    s.next()
    assert p.pin(p.block_table(first), 32)
    second = s.submit(
        list(range(32)), 8, [], shared_blocks=p.block_table(first), shared_tokens=32
    )
    assert p.committed_blocks() == 4
    s.cancel(first)
    s.cancel(second)
    assert p.stats()["reserved_blocks"] == 0
    invariant(p)


@pytest.mark.parametrize("seed", [3, 71, 909])
def test_randomized_reference_and_capacity_invariants(seed):
    rng = random.Random(seed)
    p = pool(16)
    snapshots = []
    next_id = 1
    for _ in range(600):
        op = rng.randrange(5)
        if op == 0:
            tokens = rng.randrange(1, 50)
            if p.reserve(next_id, tokens):
                p.ensure_tokens(next_id, tokens)
            next_id += 1
        elif op == 1 and p.allocations:
            request = rng.choice(list(p.allocations))
            allocation = p.allocations[request]
            table = p.block_table(request)
            if p.pin(table, allocation.live_tokens):
                snapshots.append((table, allocation.live_tokens))
        elif op == 2 and p.allocations:
            p.release(rng.choice(list(p.allocations)))
        elif op == 3 and snapshots:
            i = rng.randrange(len(snapshots))
            table, tokens = snapshots.pop(i)
            p.unpin(table, tokens)
        elif op == 4 and snapshots:
            table, tokens = rng.choice(snapshots)
            if p.reserve_shared(next_id, tokens + 1, table, tokens):
                p.ensure_tokens(next_id, tokens + 1)
                p.make_writable(next_id, tokens, 1, lambda *a: None)
            next_id += 1
        invariant(p)
    for request in list(p.allocations):
        p.release(request)
    for table, tokens in snapshots:
        p.unpin(table, tokens)
    invariant(p)
    assert p.stats()["allocated_blocks"] == p.stats()["reserved_blocks"] == 0
