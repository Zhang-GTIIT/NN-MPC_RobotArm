"""Canonical causal variants for fixed-delay residual MPC experiments."""

from __future__ import annotations

from dataclasses import dataclass


PROTOCOL_NAMES = (
    "full",
    "naive_delayed",
    "no_future_alignment",
    "no_reanchor",
    "no_feedback",
)


@dataclass(frozen=True)
class DelayProtocol:
    name: str
    future_state: bool
    future_reference: bool
    reanchor_residual: bool
    feedback: bool

    @property
    def replay_absolute(self) -> bool:
        return not self.reanchor_residual


_PROTOCOLS = {
    "full": DelayProtocol("full", True, True, True, True),
    "naive_delayed": DelayProtocol("naive_delayed", False, False, False, False),
    # State forecast and activation-time reference shift are one future-
    # alignment module in the paper.  This variant removes both together.
    "no_future_alignment": DelayProtocol("no_future_alignment", False, False, True, True),
    "no_reanchor": DelayProtocol("no_reanchor", True, True, False, True),
    "no_feedback": DelayProtocol("no_feedback", True, True, True, False),
}


def resolve_delay_protocol(name: str) -> DelayProtocol:
    try:
        return _PROTOCOLS[str(name)]
    except KeyError as exc:
        raise ValueError(f"delay_protocol must be one of {PROTOCOL_NAMES}, got {name!r}") from exc
