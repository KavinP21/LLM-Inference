# Prefix caching

Opt-in reuse of computed prompt prefixes in the MLX engine. Agent loops resend
the whole conversation every step, and sub-agents often share a system prompt,
so most of each prompt has been computed before. With the default
`prefix_cache_bytes=0` nothing is kept.

```python
from instinct import MlxEngine

with MlxEngine(
    "models/qwen2.5-0.5b.engine",
    kv_cache_bytes=2 << 30,
    prefix_cache_bytes=1 << 30,  # part of kv_cache_bytes, not on top of it
) as engine:
    system = [17, 23, 42] * 100
    reply = engine.generate(system + [5, 6], 32, checkpoints=[len(system)])
    # The next step of the same conversation reuses everything up to here...
    engine.generate(system + [5, 6] + reply[:-1] + [7], 32)
    # ...and a sibling with the same system prompt reuses the checkpoint.
    engine.generate(system + [9], 32)
    print(engine.stats()["prefix_cache"])
```

## What an entry is

An entry is a token prefix whose K/V pages stay pinned by reference count after
the request that wrote them has finished, plus a copy of any recurrent state at
exactly that length. Entries are made

- when a request finishes, covering its prompt and every reply token whose K/V
  was written (all but the last), and
- at `checkpoints`: prompt lengths the caller asks for, such as the end of a
  shared system prompt. Prefill stops a chunk there so the boundary is exact.

A new request uses the longest entry that is a proper prefix of its prompt. Its
page tables start as the entry's tables with every page's reference count
raised; a partial last page is copied first, since the request will append to
it. Entries match only at their exact length. Sliding-window layers have already
released everything before their last window, and recurrent state exists only
for the final position, so a prefix can't be resumed from the middle.

## Limits

`cache_namespace` keeps entries apart (all namespaces share one budget); it's
not an access-control mechanism. Entries from one conversation share most of
their pages, and `prefix_cache_bytes` counts each page once.

When admission needs pages, or the cache is over `prefix_cache_bytes` or
`prefix_cache_entries`, entries are dropped in this order:

1. Replies that a later reply of the same conversation extends. Each step of an
   agent loop supersedes the one before it.
2. Everything else, least recently used first.

Checkpoints are where the next chat turn or a sibling sub-agent branches off. A
long tool loop therefore no longer pushes them out. If dropping every entry
still wouldn't free enough pages, the pages belong to running requests, so
nothing is dropped and the new request waits. Dropping an entry only releases
its references, so a running request that attached it keeps its pages.

The engine isn't thread-safe; callers serialize access. Nothing is shared
across engines or kept on disk, and the CUDA engine has no prefix cache.
