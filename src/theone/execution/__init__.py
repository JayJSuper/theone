"""Auditable safe execution — The One's boundary for acting on the world.

Adopts the solid part of a sandbox (path containment, command denylist, dry-run)
and adds what makes it *The One's*: every action emits an auditable execution
credential, and any action *driven by a causal recommendation* is gated on that
recommendation being both independently-recomputable AND admissible (the two
orthogonal gates of os_loop_constrained). Only verifiably-safe and verifiably-
admissible causal advice gets to touch the world.
"""
from .safe_executor import SafeExecutor, ExecutionCredential

__all__ = ["SafeExecutor", "ExecutionCredential"]
