"""Tests for numeric comparison guards (Extension 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from nemoir_runtime.errors import (
    PolicyEvaluationError,
    StageOutputValidationError,
)
from nemoir_runtime.runtime import (
    ExprSpec,
    GuardSpec,
    InputSpec,
    RefSpec,
    StageContext,
    StageSpec,
    TransitionSpec,
    WorkflowManifest,
    WorkflowRuntime,
    WriteSpec,
)
from nemoir_runtime.tools import ToolRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping


def _make_manifest(
    *,
    stages: tuple[StageSpec, ...],
    inputs: tuple[InputSpec, ...] = (InputSpec(name="eps", type="number"),),
    entry_id: str = "A",
    exit_ids: frozenset[str] = frozenset({"B"}),
) -> WorkflowManifest:
    return WorkflowManifest(
        workflow_id="Test",
        entry_stage_id=entry_id,
        exit_stage_ids=exit_ids,
        inputs=inputs,
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )


def _make_tool_registry() -> ToolRegistry:
    return ToolRegistry([])


def _scripted_executor(outputs_by_stage: Mapping[str, list[Mapping[str, object]]]) -> Any:
    class ScriptedExecutor:
        def __init__(self) -> None:
            self.outputs: dict[str, list[Mapping[str, object]]] = {
                k: list(v) for k, v in outputs_by_stage.items()
            }

        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            bucket = self.outputs.get(ctx.stage.id, [])
            if not bucket:
                msg = f"No scripted output for stage '{ctx.stage.id}'"
                raise RuntimeError(msg)
            return bucket.pop(0)

    return ScriptedExecutor()


# ------------------------------------------------------------------
# Compare / BinOp evaluation
# ------------------------------------------------------------------


async def test_compare_gt_guard_matches() -> None:
    """A Compare(gt) guard on a numeric output matches when the value is greater."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="gt",
                            left=ExprSpec(
                                kind="ref", ref=RefSpec(kind="node_output", node="A", field="score")
                            ),
                            right=ExprSpec(kind="literal", type="number", value=5.0),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 10.0}], "B": [{"out": "done"}]}),
    )
    result = await runtime.run({"eps": 0.05})
    assert result.output["out"] == "done"


async def test_compare_guard_fails_when_below() -> None:
    """A Compare(gt) guard fails when the value is not greater, causing NoTransitionMatched."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="gt",
                            left=ExprSpec(
                                kind="ref", ref=RefSpec(kind="node_output", node="A", field="score")
                            ),
                            right=ExprSpec(kind="literal", type="number", value=5.0),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 3.0}], "B": [{"out": "done"}]}),
    )
    # 3.0 > 5.0 → False → no transition matched
    with pytest.raises(Exception):
        await runtime.run({"eps": 0.05})


async def test_binop_sub_and_compare() -> None:
    """BinOp(sub) + Compare(gt): score - Best.score > eps."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="gt",
                            left=ExprSpec(
                                kind="binop",
                                op="sub",
                                left=ExprSpec(
                                    kind="ref",
                                    ref=RefSpec(kind="node_output", node="A", field="score"),
                                ),
                                right=ExprSpec(kind="literal", type="number", value=3.0),
                            ),
                            right=ExprSpec(kind="ref", ref=RefSpec(kind="input", name="eps")),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 10.0}], "B": [{"out": "done"}]}),
    )
    # 10.0 - 3.0 = 7.0 > 0.05 → True
    result = await runtime.run({"eps": 0.05})
    assert result.output["out"] == "done"


# ------------------------------------------------------------------
# None propagation and crash-trial semantics
# ------------------------------------------------------------------


async def test_compare_none_operand_returns_false() -> None:
    """Compare with a None operand returns False (fail-closed)."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=True),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="gt",
                            left=ExprSpec(
                                kind="ref", ref=RefSpec(kind="node_output", node="A", field="score")
                            ),
                            right=ExprSpec(kind="literal", type="number", value=0.0),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": None}], "B": [{"out": "done"}]}),
    )
    # None score → compare returns False → no transition matched → error
    with pytest.raises(Exception):
        await runtime.run({"eps": 0.05})


async def test_binop_none_operand_returns_none() -> None:
    """BinOp with a None operand returns None (SQL-null propagation)."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=True),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="gt",
                            left=ExprSpec(
                                kind="binop",
                                op="sub",
                                left=ExprSpec(
                                    kind="ref",
                                    ref=RefSpec(kind="node_output", node="A", field="score"),
                                ),
                                right=ExprSpec(kind="literal", type="number", value=0.5),
                            ),
                            right=ExprSpec(kind="literal", type="number", value=0.0),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": None}], "B": [{"out": "done"}]}),
    )
    # score - 0.5 → None (binop None propagation)
    # None > 0.0 → False (compare None propagation) → no match
    with pytest.raises(Exception):
        await runtime.run({"eps": 0.05})


# ------------------------------------------------------------------
# Division-by-zero
# ------------------------------------------------------------------


async def test_div_by_zero_raises() -> None:
    """Div by zero raises PolicyEvaluationError."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="gt",
                            left=ExprSpec(
                                kind="binop",
                                op="div",
                                left=ExprSpec(
                                    kind="ref",
                                    ref=RefSpec(kind="node_output", node="A", field="score"),
                                ),
                                right=ExprSpec(kind="literal", type="number", value=0),
                            ),
                            right=ExprSpec(kind="literal", type="number", value=0.0),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 10.0}], "B": [{"out": "done"}]}),
    )
    with pytest.raises(PolicyEvaluationError, match="Division by zero"):
        await runtime.run({"eps": 0.05})


# ------------------------------------------------------------------
# Defensive op check
# ------------------------------------------------------------------


async def test_compare_defensive_op_check() -> None:
    """Hand-built ExprSpec with compare op='eq' raises at runtime (defensive check)."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="eq",  # forbidden op
                            left=ExprSpec(
                                kind="ref", ref=RefSpec(kind="node_output", node="A", field="score")
                            ),
                            right=ExprSpec(kind="literal", type="number", value=5.0),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 5.0}], "B": [{"out": "done"}]}),
    )
    with pytest.raises(PolicyEvaluationError, match="Unknown compare op"):
        await runtime.run({"eps": 0.05})


# ------------------------------------------------------------------
# Number write validation (bool rejection)
# ------------------------------------------------------------------


async def test_bool_rejected_for_number_write() -> None:
    """A bool output rejected when stage writes type=number."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(kind="always"),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": True}], "B": [{"out": "done"}]}),
    )
    with pytest.raises(StageOutputValidationError, match="expected int/float"):
        await runtime.run({"eps": 0.05})


# ------------------------------------------------------------------
# Compound guards (and/or)
# ------------------------------------------------------------------


async def test_compound_and_guard() -> None:
    """Compound And guard: both conditions must be true."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="and",
                            exprs=(
                                ExprSpec(
                                    kind="compare",
                                    op="gt",
                                    left=ExprSpec(
                                        kind="ref",
                                        ref=RefSpec(kind="node_output", node="A", field="score"),
                                    ),
                                    right=ExprSpec(kind="literal", type="number", value=5.0),
                                ),
                                ExprSpec(
                                    kind="compare",
                                    op="lt",
                                    left=ExprSpec(
                                        kind="ref",
                                        ref=RefSpec(kind="node_output", node="A", field="score"),
                                    ),
                                    right=ExprSpec(kind="literal", type="number", value=15.0),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 10.0}], "B": [{"out": "done"}]}),
    )
    result = await runtime.run({"eps": 0.05})
    assert result.output["out"] == "done"


# ------------------------------------------------------------------
# Literal round-trip
# ------------------------------------------------------------------


async def test_numeric_literal_round_trip() -> None:
    """A numeric literal (0.05) round-trips to the same float value."""
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="if",
                        cond=ExprSpec(
                            kind="compare",
                            op="gt",
                            left=ExprSpec(
                                kind="ref", ref=RefSpec(kind="node_output", node="A", field="score")
                            ),
                            right=ExprSpec(kind="literal", type="number", value=0.05),
                        ),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 0.1}], "B": [{"out": "done"}]}),
    )
    result = await runtime.run({"eps": 0.05})
    assert result.output["out"] == "done"


# ------------------------------------------------------------------
# Defensive: GuardSpec(kind="eq") rejects numeric operands at runtime
# ------------------------------------------------------------------


async def test_guard_eq_rejects_numeric_operands() -> None:
    """GuardSpec(kind='eq') with numeric operands raises PolicyEvaluationError.

    Per plan §3.4, the ordering-only rule is load-bearing. Even though the DSL
    no longer emits Guard::Eq after the §3.5 desugar, runtime manifests can be
    constructed directly and must be defended against numeric equality.
    """
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="score", type="number", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="explicit_transition",
                    guard=GuardSpec(
                        kind="eq",
                        left=ExprSpec(kind="literal", type="number", value=1.0),
                        right=ExprSpec(kind="literal", type="number", value=1),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"score": 5.0}], "B": [{"out": "done"}]}),
    )
    with pytest.raises(
        PolicyEvaluationError,
        match="does not support number operands",
    ):
        await runtime.run({"eps": 0.05})
