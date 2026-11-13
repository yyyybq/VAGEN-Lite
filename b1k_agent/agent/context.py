"""Context variables set by the server around each loop step, so infrastructure (provider call
logging) can attribute work to the right episode and slot without modules passing ids around."""

from __future__ import annotations

import contextvars

current_episode: contextvars.ContextVar[str] = contextvars.ContextVar("current_episode", default="")
current_slot: contextvars.ContextVar[str] = contextvars.ContextVar("current_slot", default="")
current_step: contextvars.ContextVar[int] = contextvars.ContextVar("current_step", default=-1)
