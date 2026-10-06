"""Versioned paired QA data and cross-interface evaluation for Active Spatial."""

from .contract import (
    ACTIVE_TASK_TYPES,
    QA_SCHEMA_VERSION,
    SUCCESS_PREDICATE_VERSION,
    build_task_contract,
    evaluate_state,
)

__all__ = [
    "ACTIVE_TASK_TYPES",
    "QA_SCHEMA_VERSION",
    "SUCCESS_PREDICATE_VERSION",
    "build_task_contract",
    "evaluate_state",
]
