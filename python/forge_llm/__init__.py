"""Python convenience layer for the Forge LLM runtime."""

try:
    from _forge import Engine, SequenceState, build_info
except ImportError:  # Host-only environments can still use export and result tooling.
    Engine = None
    SequenceState = None
    build_info = lambda: {}

__all__ = ["Engine", "SequenceState", "build_info"]
