"""Dream-RSI: Recursive Self-Improvement through Evolving Worlds for `MCL4-COE`.

Implements the complete Dream-RSI loop (Zheng et al., 2026, Section 3 & Eq. 1) natively
integrated with `MCL4-COE`'s 9 Cognitive Operators and `Ornith-1.5`:
1. Prefix-Observable Discovery Tree & Exact Replay Simulator (`Node`, `World`, `View`,
   `Outcome`, `OnlineResult`, `ReplayResult`, `online_rollout`, `replay`, `score_policy`):
   - Online exploration deploys policy $\\pi_t$ to expand a discovery tree $\\mathcal{T}_t$
     where the root opens new branches (`CONTINUE(0)`) and each non-root node has at most
     one recorded continuation (`CONTINUE(v)`).
   - Historical replay (`replay`, `score_policy`) evaluates candidate exploration policies
     against the growing world archive $\\mathcal{H}_t = (\\mathcal{T}_1, \\dots, \\mathcal{T}_t)$
     across multi-budget preferences $\\rho \\in (0, 0.25, 0.5, 0.75, 1.0)$ using normalized
     Pareto hypervolume minus round/call penalty at **zero real execution cost**.
2. Executable Policy Language & Monotonic Policy Evolution (`Program`, `GRAMMAR`,
   `RANK_FEATURES`, `improve_policy`, `DreamState`):
   - Stateless 6-directive exploration programs (`WIDTH`, `ROOT`, `RANK`, `STOP`,
     `PATIENCE`, `ROUNDS`) ranking causally visible frontier leaves by `SCORE`, `GAIN`,
     `DEEP`, `SHALLOW`, `RECENT`, `OLD`, plus `MCL4-COE` operator signals `UTILITY`
     (`Evaluation`), `NOVELTY` (`Association`), and `CONFIDENCE` (`Metacognition`).
   - Because the incumbent $\\pi^0 = \\pi_t$ is always scored alongside revisions
     $\\pi^1, \\dots, \\pi^{M}$, the selected policy $\\pi_{t+1}$ is guaranteed never to
     score worse than $\\pi_t$ on $\\mathcal{H}_t$.
3. Native `MCL4-COE` Discovery & Dreaming Engine (`MCL4DreamDiscovery`,
   `MCL4PolicyProposer`, `DreamRSIEngine`, `DreamCycleOutput`):
   - Connects `MCL4-COE`'s 9 Cognitive Operators (`Abduction`, `Association`, `Deduction`,
     `Evaluation`, `Metacognition`, `Decision`) to online tree expansion and offline
     policy-revision synthesis, with atomic JSON state persistence for Web UI & CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Strict JSON & Numeric Validators
# ---------------------------------------------------------------------------


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _json(value: Any, name: str = "payload") -> Any:
    """Copy and validate JSON without accepting NaNs or arbitrary objects."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return _number(value, name)
    if type(value) is list:
        return [_json(item, name) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json(item, name) for key, item in value.items()}
    raise ValueError(f"{name} must contain only JSON values with string keys")


def _dictionary(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return _json(value, name)


def _keys(value: Any, names: set[str], name: str) -> dict[str, Any]:
    value = _dictionary(value, name)
    if set(value) != names:
        raise ValueError(f"{name} fields must be exactly {sorted(names)}")
    return value


# ---------------------------------------------------------------------------
# Prefix-Observable Discovery Tree & Historical Replay Simulator (Section 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """One node in a Dream-RSI discovery tree $\\mathcal{T}_t$."""

    id: int
    parent: int | None
    score: float
    observation: dict[str, Any]

    def __post_init__(self) -> None:
        _integer(self.id, "node id")
        if self.parent is not None:
            _integer(self.parent, "parent id")
        object.__setattr__(self, "score", _number(self.score, "score"))
        object.__setattr__(self, "observation", _dictionary(self.observation, "observation"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent,
            "score": self.score,
            "observation": _dictionary(self.observation, "observation"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        return cls(**_keys(data, {"id", "parent", "score", "observation"}, "node"))


def _nodes(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    if type(nodes) is not tuple or not nodes or any(type(node) is not Node for node in nodes):
        raise ValueError("nodes must be a nonempty tuple of Node records")
    copied = tuple(Node.from_dict(node.to_dict()) for node in nodes)
    root = copied[0]
    if root.id != 0 or root.parent is not None or root.score != 0:
        raise ValueError("first node must be root id 0, parent None, score 0")
    seen = {0}
    extended: set[int] = set()
    for node in copied[1:]:
        if node.id in seen:
            raise ValueError("node ids must be unique")
        if node.parent not in seen:
            raise ValueError("each non-root node must reference an earlier parent")
        if node.parent != 0:
            if node.parent in extended:
                raise ValueError("each non-root node may have only one recorded child")
            extended.add(node.parent)
        seen.add(node.id)
    return copied


@dataclass(frozen=True)
class World:
    """Recorded discovery tree $\\mathcal{T}_t$ acting as an exact replay simulator."""

    world_id: str
    nodes: tuple[Node, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.world_id) is not str or not self.world_id.strip():
            raise ValueError("world_id must be a nonempty string")
        object.__setattr__(self, "nodes", _nodes(self.nodes))
        object.__setattr__(self, "metadata", _dictionary(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": _dictionary(self.metadata, "metadata"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> World:
        data = _keys(data, {"world_id", "nodes", "metadata"}, "world")
        if type(data["nodes"]) is not list:
            raise ValueError("serialized nodes must be a list")
        return cls(
            data["world_id"],
            tuple(Node.from_dict(node) for node in data["nodes"]),
            data["metadata"],
        )


@dataclass(frozen=True)
class View:
    """Causally revealed prefix of a discovery tree visible to an exploration policy."""

    nodes: tuple[Node, ...]
    round_index: int
    max_rounds: int
    max_calls: int
    workers: int
    rho: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _nodes(self.nodes))
        _integer(self.round_index, "round_index")
        _integer(self.max_rounds, "max_rounds")
        _integer(self.max_calls, "max_calls")
        _integer(self.workers, "workers", 1)
        rho = _number(self.rho, "rho")
        if not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0, 1]")
        if self.round_index > self.max_rounds or self.calls > self.max_calls:
            raise ValueError("view exceeds its budget")
        object.__setattr__(self, "rho", rho)

    @property
    def calls(self) -> int:
        return len(self.nodes) - 1

    @property
    def eligible(self) -> tuple[int, ...]:
        parents = {node.parent for node in self.nodes[1:]}
        return (0,) + tuple(node.id for node in self.nodes[1:] if node.id not in parents)


@dataclass(frozen=True)
class Outcome:
    """Result of one online discovery evaluation (`CONTINUE(parent_id)`)."""

    score: float
    observation: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _number(self.score, "score"))
        object.__setattr__(self, "observation", _dictionary(self.observation, "observation"))


Policy = Callable[[View], Sequence[int]]
Discover = Callable[[View, int, int], Outcome]


class DiscoveryStopped(RuntimeError):
    """Cooperative cancellation of an unfinished discovery or policy proposal."""


def _check_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None:
        if not callable(should_stop):
            raise ValueError("should_stop must be callable")
        if should_stop():
            raise DiscoveryStopped("Dream-RSI historical replay stopped")


@dataclass(frozen=True)
class OnlineResult:
    """Result of an online discovery tree rollout under policy $\\pi_t$."""

    world: World
    rounds: int
    agent_calls: int
    actions: tuple[tuple[int, ...], ...]
    stopped: bool = False


@dataclass(frozen=True)
class ReplayResult:
    """Result of replaying a candidate policy over one recorded `World` at zero execution cost."""

    calls: int
    rounds: int
    best_score: float
    actions: tuple[tuple[int, ...], ...]
    revealed_ids: tuple[int, ...]
    exhausted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "rounds": self.rounds,
            "best_score": self.best_score,
            "actions": [list(action) for action in self.actions],
            "revealed_ids": list(self.revealed_ids),
            "exhausted": self.exhausted,
        }


def _view(
    nodes: Sequence[Node],
    round_index: int,
    max_rounds: int,
    max_calls: int,
    workers: int,
    rho: float,
) -> View:
    return View(tuple(nodes), round_index, max_rounds, max_calls, workers, rho)


def _fast_view(
    nodes: Sequence[Node],
    round_index: int,
    max_rounds: int,
    max_calls: int,
    workers: int,
    rho: float,
) -> View:
    v = object.__new__(View)
    object.__setattr__(v, "nodes", nodes if isinstance(nodes, tuple) else tuple(nodes))
    object.__setattr__(v, "round_index", round_index)
    object.__setattr__(v, "max_rounds", max_rounds)
    object.__setattr__(v, "max_calls", max_calls)
    object.__setattr__(v, "workers", workers)
    object.__setattr__(v, "rho", float(rho))
    return v


def _action(policy: Policy, view: View) -> tuple[int, ...]:
    arg_view = (
        _fast_view(view.nodes, view.round_index, view.max_rounds, view.max_calls, view.workers, view.rho)
        if isinstance(policy, Program)
        else _view(view.nodes, view.round_index, view.max_rounds, view.max_calls, view.workers, view.rho)
    )
    answer = policy(arg_view)
    if not isinstance(answer, Sequence) or isinstance(answer, (str, bytes)):
        raise ValueError("policy must return a sequence of eligible node ids")
    action = tuple(answer)
    if any(type(node_id) is not int for node_id in action):
        raise ValueError("policy actions must be integer node ids")
    if len(action) != len(set(action)):
        raise ValueError("policy actions must not contain duplicate nodes")
    if len(action) > view.workers:
        raise ValueError("policy batch exceeds worker limit")
    if len(action) > view.max_calls - view.calls:
        raise ValueError("policy batch exceeds remaining call budget")
    if not set(action).issubset(view.eligible):
        raise ValueError("policy selected a node outside the visible eligible frontier")
    return action


def online_rollout(
    policy: Policy,
    discover: Discover,
    *,
    world_id: str,
    root_observation: dict[str, Any],
    metadata: dict[str, Any],
    max_rounds: int,
    workers: int,
    max_calls: int,
    rho: float = 0.5,
    should_stop: Callable[[], bool] | None = None,
) -> OnlineResult:
    """Deploy `policy` online to expand a discovery tree `World` ($\\mathcal{T}_t$)."""
    if should_stop is not None and not callable(should_stop):
        raise ValueError("should_stop must be callable")
    initial = World(world_id, (Node(0, None, 0.0, root_observation),), metadata)
    nodes = list(initial.nodes)
    actions: list[tuple[int, ...]] = []
    stopped = False
    view = _view(nodes, 0, max_rounds, max_calls, workers, rho)
    while view.round_index < max_rounds and view.calls < max_calls:
        if should_stop is not None and should_stop():
            stopped = True
            break
        action = _action(policy, view)
        if not action:
            break
        additions = []
        executed = []
        if len(action) > 1 and hasattr(discover, "discover_batch") and should_stop is None:
            try:
                outcomes = discover.discover_batch(
                    _fast_view(
                        view.nodes,
                        view.round_index,
                        max_rounds,
                        max_calls,
                        workers,
                        rho,
                    ),
                    action,
                    len(nodes),
                )
                for idx_a, (parent_id, outcome) in enumerate(zip(action, outcomes)):
                    if type(outcome) is not Outcome:
                        raise ValueError("discover must return an Outcome")
                    node_id = len(nodes) + idx_a
                    additions.append(Node(node_id, parent_id, outcome.score, outcome.observation))
                    executed.append(parent_id)
            except DiscoveryStopped:
                stopped = True
        else:
            for parent_id in action:
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                node_id = len(nodes) + len(additions)
                try:
                    outcome = discover(
                        _fast_view(
                            view.nodes,
                            view.round_index,
                            max_rounds,
                            max_calls,
                            workers,
                            rho,
                        )
                        if hasattr(discover, "discover_batch")
                        else _view(
                            view.nodes,
                            view.round_index,
                            max_rounds,
                            max_calls,
                            workers,
                            rho,
                        ),
                        parent_id,
                        node_id,
                    )
                except DiscoveryStopped:
                    stopped = True
                    break
                if type(outcome) is not Outcome:
                    raise ValueError("discover must return an Outcome")
                additions.append(Node(node_id, parent_id, outcome.score, outcome.observation))
                executed.append(parent_id)
        nodes.extend(additions)
        if executed:
            actions.append(tuple(executed))
        view = _fast_view(nodes, len(actions), max_rounds, max_calls, workers, rho)
        if stopped:
            break
    world = World(world_id, tuple(nodes), initial.metadata)
    return OnlineResult(world, len(actions), len(nodes) - 1, tuple(actions), stopped)


def replay(
    world: World,
    policy: Policy,
    *,
    rho: float,
    max_rounds: int,
    workers: int,
    max_calls: int,
    should_stop: Callable[[], bool] | None = None,
) -> ReplayResult:
    """Replay `policy` over a recorded `World` at zero model/verifier execution cost."""
    _check_stop(should_stop)
    if not (type(world) is World and isinstance(policy, Program)):
        world = World.from_dict(world.to_dict())
    children: dict[int, list[Node]] = {}
    for node in world.nodes[1:]:
        children.setdefault(node.parent, []).append(node)
    revealed = [world.nodes[0]]
    seen = {0}
    actions: list[tuple[int, ...]] = []
    exhausted = 0
    view = _fast_view(revealed, 0, max_rounds, max_calls, workers, rho)
    while (
        view.round_index < max_rounds
        and view.calls < max_calls
        and len(seen) < len(world.nodes)
    ):
        _check_stop(should_stop)
        action = _action(policy, view)
        if not action:
            break
        additions = []
        for parent_id in action:
            child = next(
                (node for node in children.get(parent_id, ()) if node.id not in seen),
                None,
            )
            if child is None:
                exhausted += 1
            else:
                additions.append(child)
                seen.add(child.id)
        revealed.extend(additions)
        actions.append(action)
        view = _fast_view(revealed, len(actions), max_rounds, max_calls, workers, rho)
    _check_stop(should_stop)
    best = max((node.score for node in revealed[1:]), default=0.0)
    return ReplayResult(
        len(revealed) - 1,
        len(actions),
        best,
        tuple(actions),
        tuple(node.id for node in revealed),
        exhausted,
    )


def score_policy(
    worlds: Sequence[World],
    policy: Policy,
    *,
    max_rounds: int,
    workers: int,
    max_calls: int,
    round_penalty: float = 0.05,
    rhos: Sequence[float] = (0, 0.25, 0.5, 0.75, 1),
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Dream-RSI Paper Eq. 1: normalized Pareto hypervolume minus round/call penalty."""
    _check_stop(should_stop)
    if not worlds:
        raise ValueError("score_policy requires a nonempty world archive")
    penalty = _number(round_penalty, "round_penalty")
    if penalty < 0:
        raise ValueError("round_penalty must be nonnegative")
    rhos = tuple(_number(rho, "rho") for rho in rhos)
    if not rhos or len(set(rhos)) != len(rhos) or any(not 0 <= rho <= 1 for rho in rhos):
        raise ValueError("rhos must be distinct values in [0, 1]")
    receipts = []
    total_replays = 0
    for original in worlds:
        _check_stop(should_stop)
        world = (
            original
            if (type(original) is World and isinstance(policy, Program))
            else World.from_dict(original.to_dict())
        )
        count = len(world.nodes) - 1
        ceiling = max((node.score for node in world.nodes[1:]), default=0.0)
        runs = []
        for rho in rhos:
            _check_stop(should_stop)
            result = replay(
                world,
                policy,
                rho=rho,
                max_rounds=max_rounds,
                workers=workers,
                max_calls=max_calls,
                should_stop=should_stop,
            )
            total_replays += 1
            attainment = (
                min(1.0, max(0.0, result.best_score / ceiling)) if ceiling > 0 else 0.0
            )
            runs.append({"rho": rho, **result.to_dict(), "attainment": attainment})
        qualities: dict[int, float] = {}
        for run in runs:
            n = run["calls"]
            qualities[n] = max(qualities.get(n, -math.inf), run["attainment"])
        frontier = []
        highest = -math.inf
        for n, quality in sorted(qualities.items()):
            if quality > highest:
                frontier.append({"calls": n, "attainment": quality})
                highest = quality
        hypervolume = 0.0
        previous = 0.0
        for point in frontier:
            if count:
                hypervolume += (
                    (point["attainment"] - previous) * (count - point["calls"]) / count
                )
            previous = point["attainment"]
        parallel_penalty = sum(
            run["rounds"] / run["calls"] if run["calls"] else 0.0 for run in runs
        ) / len(runs)
        objective = hypervolume - penalty * parallel_penalty
        receipts.append(
            {
                "world_id": world.world_id,
                "recorded_calls": count,
                "recorded_best": ceiling,
                "positive_score_support": ceiling > 0,
                "objective": objective,
                "hypervolume": hypervolume,
                "parallel_penalty": parallel_penalty,
                "frontier": frontier,
                "replays": runs,
            }
        )
    _check_stop(should_stop)
    return {
        "mean_objective": sum(item["objective"] for item in receipts) / len(receipts),
        "round_penalty": penalty,
        "rhos": list(rhos),
        "replays_evaluated": total_replays,
        "worlds": receipts,
    }


# ---------------------------------------------------------------------------
# Executable Exploration Policy DSL & Monotonic Offline Evolution
# ---------------------------------------------------------------------------

MAX_POLICY_BYTES = 512
MAX_INTEGER = 1_000_000
MAX_FEEDBACK_RHOS = 16
RANK_FEATURES = (
    "SCORE",
    "GAIN",
    "DEEP",
    "SHALLOW",
    "RECENT",
    "OLD",
    "UTILITY",
    "NOVELTY",
    "CONFIDENCE",
    "PP",
    "RECURSION",
)

GRAMMAR = (
    "Write exactly six lines in this order, without prose or code fences:\n"
    "WIDTH n\nROOT n\nRANK feature [feature ...]\nSTOP NEVER|number\nPATIENCE n\nROUNDS n\n"
    "WIDTH and ROUNDS are positive integers. ROOT and PATIENCE are nonnegative integers; "
    "all integers are at most 1000000. ROOT 0 opens only the initial branch; ROOT n>0 "
    "opens a fresh branch every nth round. RANK is a nonempty ordered list of unique "
    "SCORE, GAIN, DEEP, SHALLOW, RECENT, OLD, UTILITY, NOVELTY, CONFIDENCE, PP, RECURSION features "
    "(first feature has priority). STOP is NEVER or a score in [0,1]; stop when an "
    "observed result reaches it. PATIENCE 0 disables patience, otherwise stop after n "
    "observed evaluations without a strict improvement in best score. WIDTH caps the "
    "simultaneous batch; ROUNDS caps decision rounds. Rho scales call and round budgets "
    "from one at rho=0 to full at rho=1. Return at most 512 UTF-8 bytes. Example:\n"
    "WIDTH 4\nROOT 1\nRANK SCORE GAIN SHALLOW\nSTOP NEVER\nPATIENCE 0\nROUNDS 16"
)


def _bounded_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or not minimum <= value <= MAX_INTEGER:
        raise ValueError(f"{name} must be an integer in {minimum}..{MAX_INTEGER}")
    return value


def _parse_integer(value: str, name: str, minimum: int) -> int:
    if re.fullmatch(r"0|[1-9][0-9]{0,6}", value) is None:
        raise ValueError(f"{name} must be a canonical nonnegative integer")
    return _bounded_int(int(value), name, minimum)


def _budget(maximum: int, rho: float) -> int:
    return 0 if maximum == 0 else 1 + math.floor(rho * (maximum - 1))


@dataclass(frozen=True)
class Program:
    """Stateless, deterministic exploration policy over a causally visible `View`."""

    width: int = 4
    root_every: int = 1
    rank: tuple[str, ...] = ("SCORE", "GAIN", "SHALLOW")
    stop: float | None = None
    patience: int = 0
    rounds: int = 16

    def __post_init__(self) -> None:
        _bounded_int(self.width, "WIDTH", 1)
        _bounded_int(self.root_every, "ROOT", 0)
        _bounded_int(self.patience, "PATIENCE", 0)
        _bounded_int(self.rounds, "ROUNDS", 1)
        if (
            type(self.rank) is not tuple
            or not self.rank
            or any(
                type(feature) is not str or feature not in RANK_FEATURES
                for feature in self.rank
            )
            or len(self.rank) != len(set(self.rank))
        ):
            raise ValueError("RANK must be a nonempty tuple of unique allowed features")
        if self.stop is not None:
            if (
                isinstance(self.stop, bool)
                or not isinstance(self.stop, (int, float))
                or not 0 <= self.stop <= 1
            ):
                raise ValueError("STOP must be None or a finite score in [0,1]")
            object.__setattr__(self, "stop", float(self.stop) if self.stop else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "root_every": self.root_every,
            "rank": list(self.rank),
            "stop": self.stop,
            "patience": self.patience,
            "rounds": self.rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Program:
        fields = {"width", "root_every", "rank", "stop", "patience", "rounds"}
        if type(data) is not dict or set(data) != fields:
            raise ValueError(f"program fields must be exactly {sorted(fields)}")
        if type(data["rank"]) is not list:
            raise ValueError("serialized RANK must be a list")
        return cls(**{**data, "rank": tuple(data["rank"])})

    def to_text(self) -> str:
        threshold = "NEVER" if self.stop is None else str(self.stop)
        return (
            f"WIDTH {self.width}\n"
            f"ROOT {self.root_every}\n"
            f"RANK {' '.join(self.rank)}\n"
            f"STOP {threshold}\n"
            f"PATIENCE {self.patience}\n"
            f"ROUNDS {self.rounds}"
        )

    @classmethod
    def from_text(cls, text: str) -> Program:
        if type(text) is not str:
            raise ValueError("program text must be a string")
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("program text must be valid UTF-8") from exc
        if size > MAX_POLICY_BYTES:
            raise ValueError(f"program text exceeds {MAX_POLICY_BYTES} bytes")
        text = text.replace("\r\n", "\n")
        if text.endswith("\n"):
            text = text[:-1]
        lines = text.split("\n")
        names = ("WIDTH", "ROOT", "RANK", "STOP", "PATIENCE", "ROUNDS")
        if len(lines) != len(names):
            raise ValueError("program must contain exactly six directive lines")
        values: dict[str, str] = {}
        for name, line in zip(names, lines):
            prefix = name + " "
            if not line.startswith(prefix):
                raise ValueError(f"expected {name} directive in canonical order")
            value = line[len(prefix) :]
            if (
                not value
                or value != value.strip()
                or "  " in value
                or any(ch.isspace() and ch != " " for ch in value)
            ):
                raise ValueError(f"invalid whitespace or empty argument for {name}")
            values[name] = value
        raw_stop = values["STOP"]
        if raw_stop == "NEVER":
            stop = None
        else:
            if (
                re.fullmatch(
                    r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", raw_stop
                )
                is None
            ):
                raise ValueError("STOP must be NEVER or a finite score in [0,1]")
            stop = float(raw_stop)
        return cls(
            width=_parse_integer(values["WIDTH"], "WIDTH", 1),
            root_every=_parse_integer(values["ROOT"], "ROOT", 0),
            rank=tuple(values["RANK"].split(" ")),
            stop=stop,
            patience=_parse_integer(values["PATIENCE"], "PATIENCE", 0),
            rounds=_parse_integer(values["ROUNDS"], "ROUNDS", 1),
        )

    def __call__(self, view: View) -> tuple[int, ...]:
        if not isinstance(view, View):
            raise ValueError("program requires a visible View, not a World")
        round_limit = _budget(min(self.rounds, view.max_rounds), view.rho)
        call_limit = _budget(view.max_calls, view.rho)
        available = min(self.width, view.workers, call_limit - view.calls)
        if view.round_index >= round_limit or available <= 0:
            return ()
        outcomes = view.nodes[1:]
        if outcomes:
            best = max(node.score for node in outcomes)
            if self.stop is not None and best >= self.stop:
                return ()
            if self.patience:
                running = -math.inf
                without_improvement = 0
                for node in outcomes:
                    if node.score > running:
                        running = node.score
                        without_improvement = 0
                    else:
                        without_improvement += 1
                if without_improvement >= self.patience:
                    return ()
        eligible = view.eligible
        chosen: list[int] = []
        if not outcomes or (self.root_every and view.round_index % self.root_every == 0):
            chosen.append(0)
        nodes = {node.id: node for node in view.nodes}
        order = {node.id: index for index, node in enumerate(view.nodes)}
        depths = {0: 0}
        for node in outcomes:
            depths[node.id] = depths[node.parent] + 1

        def _obs_float(node: Node, key: str, default: float = 0.0) -> float:
            val = node.observation.get(key, default)
            return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else default

        def rank_key(node_id: int) -> tuple[Any, ...]:
            node = nodes[node_id]
            features = {
                "SCORE": -node.score,
                "GAIN": -(node.score - nodes[node.parent].score),
                "DEEP": -depths[node_id],
                "SHALLOW": depths[node_id],
                "RECENT": -order[node_id],
                "OLD": order[node_id],
                "UTILITY": -_obs_float(node, "utility", node.score),
                "NOVELTY": -_obs_float(node, "novelty", 0.0),
                "CONFIDENCE": -_obs_float(node, "confidence", node.score),
                "PP": -_obs_float(node, "pp_weighted", node.score * 100.0),
                "RECURSION": -_obs_float(
                    node, "recursion_progress", float(depths[node_id]) * node.score
                ),
            }
            return tuple(features[feature] for feature in self.rank) + (order[node_id],)

        leaves = sorted((node_id for node_id in eligible if node_id != 0), key=rank_key)
        chosen.extend(leaves[: max(0, available - len(chosen))])
        return tuple(chosen)


Propose = Callable[[Program, dict[str, Any], int], dict[str, Any]]


def _feedback(
    program: Program,
    score: float,
    report: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    revision: int,
    max_rounds: int,
    workers: int,
    max_calls: int,
) -> dict[str, Any]:
    """Construct bounded replay diagnostics without leaking hidden node payloads."""
    worlds = report["worlds"]
    rho_summary = []
    for index, rho in enumerate(report["rhos"][:MAX_FEEDBACK_RHOS]):
        runs = [world["replays"][index] for world in worlds]
        rho_summary.append(
            {
                "rho": rho,
                **{
                    "mean_" + key: sum(run[key] for run in runs) / len(runs)
                    for key in ("calls", "rounds", "attainment", "exhausted")
                },
            }
        )
    return {
        "revision": revision,
        "objective": "Maximize mean normalized quality/call Pareto hypervolume minus round penalty.",
        "current_score": score,
        "incumbent_score": score,
        "world_summary": {
            "count": len(worlds),
            "recorded_calls": sum(item["recorded_calls"] for item in worlds),
            "positive_worlds": sum(item["positive_score_support"] for item in worlds),
            "mean_hypervolume": sum(item["hypervolume"] for item in worlds) / len(worlds),
            "mean_parallel_penalty": sum(item["parallel_penalty"] for item in worlds)
            / len(worlds),
        },
        "rho_summary": rho_summary,
        "rho_summary_truncated": len(report["rhos"]) > MAX_FEEDBACK_RHOS,
        "budgets": {"max_rounds": max_rounds, "workers": workers, "max_calls": max_calls},
        "rhos": list(report["rhos"][:MAX_FEEDBACK_RHOS]),
        "round_penalty": report["round_penalty"],
        "previous_candidate": None
        if previous is None
        else {
            key: previous[key]
            for key in ("revision", "program_text", "valid", "score", "accepted", "error")
        },
    }


def improve_policy(
    worlds: Sequence[World],
    incumbent: Program,
    propose: Propose,
    *,
    revisions: int,
    max_rounds: int,
    workers: int,
    max_calls: int,
    round_penalty: float = 0.05,
    rhos: Sequence[float] = (0, 0.25, 0.5, 0.75, 1),
    should_stop: Callable[[], bool] | None = None,
    revision_offset: int = 0,
    previous_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dream offline over `worlds` ($\\mathcal{H}_t$), retaining strict replay improvements.

    Since $\\pi^0 = \\text{incumbent}$ is always evaluated first, the winning policy
    $\\pi_{t+1}$ satisfies $J(\\pi_{t+1}, \\mathcal{H}_t) \\ge J(\\pi_t, \\mathcal{H}_t)$.
    """
    if not isinstance(incumbent, Program):
        raise ValueError("incumbent must be a Program")
    _integer(revisions, "revisions", 0)
    _integer(revision_offset, "revision_offset", 0)
    if not callable(propose):
        raise ValueError("propose must be callable")
    if should_stop is not None and not callable(should_stop):
        raise ValueError("should_stop must be callable")

    copied_worlds = []
    for world in worlds:
        if should_stop is not None and should_stop():
            raise DiscoveryStopped("Dream-RSI historical replay stopped")
        copied_worlds.append(world if type(world) is World else World.from_dict(world.to_dict()))
    worlds = tuple(copied_worlds)
    rhos = tuple(rhos)
    options = dict(
        max_rounds=max_rounds,
        workers=workers,
        max_calls=max_calls,
        round_penalty=round_penalty,
        rhos=rhos,
        should_stop=should_stop,
    )
    best_report = score_policy(worlds, incumbent, **options)
    initial_report = best_report
    initial_score = best_report["mean_objective"]
    best_score = initial_score
    program = incumbent
    candidates: list[dict[str, Any]] = []
    accepted_revision = 0
    stopped = False
    replays_dreamed = int(best_report.get("replays_evaluated", len(worlds) * len(rhos)))

    for revision in range(revision_offset + 1, revision_offset + revisions + 1):
        if should_stop is not None and should_stop():
            stopped = True
            break
        feedback = _feedback(
            program,
            best_score,
            best_report,
            candidates[-1] if candidates else previous_candidate,
            revision=revision,
            max_rounds=max_rounds,
            workers=workers,
            max_calls=max_calls,
        )
        try:
            proposal = propose(program, feedback, revision)
        except DiscoveryStopped:
            stopped = True
            break
        receipt: dict[str, Any] = {
            "revision": revision,
            "program_text": None,
            "program": None,
            "finished": False,
            "valid": False,
            "score": None,
            "accepted": False,
            "error": None,
            "report": None,
        }
        try:
            if type(proposal) is not dict:
                raise ValueError("proposal must be a dictionary with text and finished")
            text = proposal.get("text")
            if type(text) is not str:
                raise ValueError("proposal text must be a string")
            receipt["program_text"] = (
                text.encode("utf-8", errors="replace")[:MAX_POLICY_BYTES].decode(
                    "utf-8", errors="replace"
                )
            )
            finished = proposal.get("finished")
            if type(finished) is not bool:
                raise ValueError("proposal finished must be boolean")
            receipt["finished"] = finished
            if not finished:
                raise ValueError("unfinished policy proposal")
            candidate = Program.from_text(text)
        except ValueError as exc:
            receipt["error"] = str(exc)
            candidates.append(receipt)
            continue
        try:
            report = score_policy(worlds, candidate, **options)
        except DiscoveryStopped:
            stopped = True
            break
        replays_dreamed += int(report.get("replays_evaluated", len(worlds) * len(rhos)))
        score = report["mean_objective"]
        receipt.update(
            program_text=candidate.to_text(),
            program=candidate.to_dict(),
            valid=True,
            score=score,
            report=report,
        )
        if score > best_score:
            program, best_score, best_report = candidate, score, report
            accepted_revision = revision
            receipt["accepted"] = True
        candidates.append(receipt)

    return {
        "program": program.to_dict(),
        "program_text": program.to_text(),
        "initial_score": initial_score,
        "final_score": best_score,
        "accepted_revision": accepted_revision,
        "replays_dreamed": replays_dreamed,
        "candidates": candidates,
        "initial_report": initial_report,
        "final_report": best_report,
        "stopped": stopped,
    }


# ---------------------------------------------------------------------------
# Bounded Archive & Atomic Persistence (`DreamState`)
# ---------------------------------------------------------------------------

STATE_VERSION = 1
MAX_HISTORY = 32


@dataclass
class DreamState:
    """Bounded world archive $\\mathcal{H}_t$ and active exploration policy $\\pi_t$."""

    program: Program = field(default_factory=Program)
    initial_program: Program = field(default_factory=Program)
    cycle: int = 0
    worlds: list[World] = field(default_factory=list)
    recent_receipts: list[dict[str, Any]] = field(default_factory=list)
    total_calls: int = 0
    total_revisions: int = 0
    accepted_revisions: int = 0
    total_replays_dreamed: int = 0
    discarded_worlds: int = 0
    initial_score: float = 0.0
    latest_score: float = 0.0

    def append_world(
        self,
        world: World,
        max_worlds: int = 64,
        max_archive_bytes: int = 4 * 1024 * 1024,
    ) -> int:
        _integer(max_worlds, "max_worlds", 1)
        _integer(max_archive_bytes, "max_archive_bytes", 2)
        if type(world) is not World:
            raise ValueError("world must be a World record")
        copied = World.from_dict(world.to_dict())
        candidates = [*self.worlds, copied]
        sizes = [
            len(json.dumps(item.to_dict(), separators=(",", ":")).encode("utf-8"))
            for item in candidates
        ]
        size = 2 + sum(sizes) + max(0, len(sizes) - 1)
        evicted = 0
        while len(candidates) - evicted > max_worlds or (
            len(candidates) - evicted > 1 and size > max_archive_bytes
        ):
            size -= sizes[evicted] + 1
            evicted += 1
        self.worlds = candidates[evicted:]
        self.discarded_worlds += evicted
        return evicted

    def add_receipt(self, receipt: dict[str, Any], max_history: int = MAX_HISTORY) -> None:
        self.recent_receipts = [*self.recent_receipts, _dictionary(receipt, "receipt")][
            -max_history:
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "program": self.program.to_dict(),
            "program_text": self.program.to_text(),
            "initial_program": self.initial_program.to_dict(),
            "cycle": self.cycle,
            "total_calls": self.total_calls,
            "total_revisions": self.total_revisions,
            "accepted_revisions": self.accepted_revisions,
            "total_replays_dreamed": self.total_replays_dreamed,
            "discarded_worlds": self.discarded_worlds,
            "initial_score": round(float(self.initial_score), 6),
            "latest_score": round(float(self.latest_score), 6),
            "worlds": [w.to_dict() for w in self.worlds],
            "recent_receipts": list(self.recent_receipts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DreamState:
        return cls(
            program=Program.from_dict(data["program"]),
            initial_program=Program.from_dict(data["initial_program"]),
            cycle=int(data.get("cycle", 0)),
            worlds=[World.from_dict(w) for w in data.get("worlds", [])],
            recent_receipts=list(data.get("recent_receipts", [])),
            total_calls=int(data.get("total_calls", 0)),
            total_revisions=int(data.get("total_revisions", 0)),
            accepted_revisions=int(data.get("accepted_revisions", 0)),
            total_replays_dreamed=int(data.get("total_replays_dreamed", 0)),
            discarded_worlds=int(data.get("discarded_worlds", 0)),
            initial_score=float(data.get("initial_score", 0.0)),
            latest_score=float(data.get("latest_score", 0.0)),
        )


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            tmp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                tmp = None
                break
            except PermissionError:
                if os.name != "nt" or attempt == 19:
                    raise
                time.sleep(0.015 * (attempt + 1))
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Native `MCL4-COE` Discovery & Policy-Revision Synthesis
# ---------------------------------------------------------------------------

_SCAFFOLD_BRANCHES: tuple[str, ...] = (
    "abduction -> association -> decision -> deduction -> evaluation -> metacognition",
    "association -> deduction -> decision -> evaluation -> metacognition",
    "abduction -> deduction -> decision -> metacognition",
    "decision -> deduction -> evaluation -> metacognition",
    "abduction -> induction -> deduction -> decision -> evaluation",
    "association -> synthesis -> deduction -> decision -> metacognition",
    "abstraction -> abduction -> deduction -> decision -> evaluation -> metacognition",
    "induction -> synthesis -> deduction -> evaluation -> decision -> metacognition",
)

_COGNITIVE_PRIMITIVES: tuple[str, ...] = (
    "abstraction",
    "abduction",
    "association",
    "synthesis",
    "induction",
    "deduction",
    "decision",
    "evaluation",
    "metacognition",
)


def synthesize_operator_scaffold(
    prev_scaffold: str,
    *,
    domain: str = "",
    depth: int = 1,
    parent_vetoed: bool = False,
    ded_entailed: float = 0.7,
    novelty: float = 0.5,
    utility: float = 0.5,
    confidence: float = 0.7,
    node_id: int = 1,
    star_difficulty: float = 1.0,
    recursion_depth: int = 1,
) -> str:
    """Dynamically synthesize or mutate a `#S$#` cognitive operator scaffold from live node telemetry."""
    ops = [op.strip() for op in prev_scaffold.split("->") if op.strip() in _COGNITIVE_PRIMITIVES]
    if not ops:
        ops = ["abduction", "deduction", "decision", "evaluation", "metacognition"]

    dom = domain.lower()
    if parent_vetoed or ded_entailed < 0.58:
        # High contradiction / veto -> inject abstraction + deduction + evaluation verification loop
        lead = ["abstraction", "deduction", "evaluation"]
    elif recursion_depth >= 3 or star_difficulty >= 3.0 or "recursion" in dom:
        # Multi-hop / infinite-horizon recursive tasks -> abstraction + abduction + induction + deduction
        lead = ["abstraction", "abduction", "induction", "deduction"]
    elif novelty < 0.35:
        # Low branch novelty -> inject abduction + association + induction hypothesis expansion
        lead = ["abduction", "association", "induction"]
    elif utility < 0.48 or confidence < 0.50:
        # Low utility/confidence -> inject synthesis + deduction before decision
        lead = ["association", "synthesis", "deduction"]
    elif "code" in dom or "python" in dom:
        lead = ["abstraction", "synthesis", "deduction"]
    elif "commonsense" in dom or "hellaswag" in dom:
        lead = ["association", "abduction", "synthesis"]
    elif "mmlu" in dom or "logic" in dom or "economics" in dom:
        lead = ["abstraction", "deduction", "evaluation"]
    elif "math" in dom or "physics" in dom or "science" in dom or "arc" in dom:
        lead = ["abduction", "induction", "deduction"]
    else:
        lead = ops[:2]

    # Combine lead with complementary tail operators without adjacent duplicates
    tail_candidates = [
        op for op in ops if op not in lead
    ] + ["deduction", "decision", "evaluation", "metacognition"]
    merged: list[str] = []
    max_ops = 6 if (recursion_depth >= 3 or star_difficulty >= 2.5) else (5 + (node_id % 2))
    for op in lead + tail_candidates:
        if op in _COGNITIVE_PRIMITIVES and op not in merged:
            merged.append(op)
        if len(merged) >= max_ops:
            break
    if "metacognition" not in merged:
        merged.append("metacognition")
    return " -> ".join(merged)


def _ancestry_chain(view: View, parent_id: int) -> list[Node]:
    nodes = {node.id: node for node in view.nodes}
    if parent_id not in nodes:
        raise ValueError(f"parent_id {parent_id} not in visible view")
    chain: list[Node] = []
    seen: set[int] = set()
    curr: int | None = parent_id
    while curr is not None:
        if curr in seen or curr not in nodes:
            raise ValueError("discovery ancestry must be rooted and acyclic")
        seen.add(curr)
        node = nodes[curr]
        chain.append(node)
        curr = node.parent
    chain.reverse()
    return chain


class MCL4DreamDiscovery:
    """Executes online discovery tree expansions (`CONTINUE(parent_id)`) using `MCL4-COE`.

    Each call to `discover(view, parent_id, node_id)`:
    1. Traces the causal ancestor chain from root `0` to `parent_id`.
    2. Selects or dynamically synthesizes a `#S$#` cognitive operator scaffold conditioned
       only on ancestor scaffolds and observed verifier/operator feedback (never the
       ground-truth `#H$#` assertion).
    3. Evaluates `coe.forward_all_operators` on the ancestor-conditioned prompt and
       candidate `#T$# -> #A$#` rollouts, recording `score`, `utility` (`Evaluation`),
       `novelty` (`Association`), and `confidence` (`Metacognition`) into the tree node.
    """

    def __init__(
        self,
        coe: Any,
        task: Any,
        *,
        max_seq_len: int = 384,
        device: str | torch.device | None = None,
    ) -> None:
        self.coe = coe
        self.task = task
        self.max_seq_len = max_seq_len
        self.device = (
            torch.device(device)
            if device is not None
            else next(coe.parameters()).device
        )
        self._node_embeddings: list[torch.Tensor] = []
        self._max_tok_len = max(self.max_seq_len, 512)
        self._vocab_size = int(getattr(self.coe.backbone.config, "vocab_size", 260))
        self._bounded_descs = tuple(
            d.encode("utf-8")[: self._max_tok_len - 4].decode("utf-8", errors="ignore")
            for d in self.task.candidate_rollouts
        )
        from .data import get_tokenizer

        self._tok = get_tokenizer(self._vocab_size)
        pad_id = self._tok.pad_id
        opt_rows: list[list[int]] = []
        opt_lens: list[int] = []
        for desc in self._bounded_descs:
            ids = self._tok.encode(desc, add_bos=True, add_eos=True)[: self._max_tok_len]
            opt_lens.append(len(ids))
            opt_rows.append(ids)
        max_o = max(opt_lens)
        self._opt_ids_1 = torch.tensor(
            [[ r + [pad_id] * (max_o - len(r)) for r in opt_rows ]],
            dtype=torch.long,
            device=self.device,
        )
        self._opt_lens_1 = torch.tensor([opt_lens], dtype=torch.long, device=self.device)
        self._targets_1 = torch.tensor(
            [self.task.gold_index], dtype=torch.long, device=self.device
        )
        self._h_scores = torch.tensor(
            self.task.harness_scores, device=self.device, dtype=torch.float32
        )
        self._cached_cand_encoded: Any | None = None

    synthesize_operator_scaffold = staticmethod(synthesize_operator_scaffold)

    def _prepare_node_context(
        self, view: View, parent_id: int, node_id: int, extra_root_children: int = 0
    ) -> tuple[int, str, list[int]]:
        ancestors = _ancestry_chain(view, parent_id)
        depth = len(ancestors) - 1
        root_children_count = (
            sum(1 for n in view.nodes[1:] if n.parent == 0) + extra_root_children
        )
        task_stars = float(getattr(self.task, "star_difficulty", 1.0))
        task_rec_depth = max(1, int(getattr(self.task, "recursion_depth", 1)))
        task_skill = str(getattr(self.task, "skill", "general.reasoning"))

        if parent_id == 0:
            scaffold_idx = root_children_count % len(_SCAFFOLD_BRANCHES)
            scaffold = _SCAFFOLD_BRANCHES[scaffold_idx]
        else:
            parent_obs = ancestors[-1].observation
            prev_scaffold = str(parent_obs.get("scaffold", _SCAFFOLD_BRANCHES[0]))
            parent_vetoed = bool(parent_obs.get("vetoed", False))
            if (node_id + depth) % 2 == 1 or parent_vetoed:
                scaffold = synthesize_operator_scaffold(
                    prev_scaffold,
                    domain=str(getattr(self.task, "domain", "")),
                    depth=depth,
                    parent_vetoed=parent_vetoed,
                    ded_entailed=float(parent_obs.get("deduction_entailed", 0.7)),
                    novelty=float(parent_obs.get("novelty", 0.5)),
                    utility=float(parent_obs.get("utility", 0.5)),
                    confidence=float(parent_obs.get("confidence", 0.7)),
                    node_id=node_id,
                    star_difficulty=task_stars,
                    recursion_depth=task_rec_depth,
                )
            else:
                base_idx = (
                    _SCAFFOLD_BRANCHES.index(prev_scaffold)
                    if prev_scaffold in _SCAFFOLD_BRANCHES
                    else 0
                )
                step_shift = 2 if parent_vetoed else 1
                scaffold = _SCAFFOLD_BRANCHES[
                    (base_idx + step_shift + depth) % len(_SCAFFOLD_BRANCHES)
                ]

        ancestor_trace = []
        for anc in ancestors[1:][-3:]:
            sc_head = str(anc.observation.get("scaffold", "")).split("->")[0].strip()[:12]
            ancestor_trace.append(
                f"[d={anc.observation.get('depth', 1)} s={anc.score:.3f} "
                f"c={anc.observation.get('confidence', 0.5):.2f} {sc_head}]"
            )
        subgoal_tag = (
            f"\n#SUBGOAL$# Hop {min(depth + 1, task_rec_depth)}/{task_rec_depth} [{task_skill} ★{task_stars:.2f}]"
            if (task_rec_depth >= 2 or task_stars >= 1.5)
            else ""
        )
        history_block = (
            "\n#H_ANC$# " + " | ".join(ancestor_trace) if ancestor_trace else ""
        ) + subgoal_tag
        max_tok_len = self._max_tok_len
        suffix = f"\n#S$# {scaffold}"
        prefix = f"{self.task.prompt_q}{history_block}"
        max_prefix_bytes = max(64, max_tok_len - len(suffix.encode("utf-8")) - 4)
        prefix_trimmed = prefix.encode("utf-8")[:max_prefix_bytes].decode(
            "utf-8", errors="ignore"
        )
        context_text = f"{prefix_trimmed}{suffix}"
        c_ids = self._tok.encode(context_text, add_bos=True, add_eos=True)[:max_tok_len]
        return depth, scaffold, c_ids

    def _ensure_cand_encoded(self) -> None:
        if self._cached_cand_encoded is not None:
            return
        from .operators import _canonical_candidate_order

        valid_options = self._opt_lens_1 > 0
        valid_lengths = self._opt_lens_1[valid_options]
        opt_width = int(valid_lengths.max().item())
        flat_opt_ids = self._opt_ids_1[valid_options][:, :opt_width].long()
        owner_rows = valid_options.nonzero(as_tuple=False)[:, 0]
        perm, _ = _canonical_candidate_order(owner_rows, valid_lengths, flat_opt_ids)
        canon_opt_ids = flat_opt_ids.index_select(0, perm)
        canon_lengths = valid_lengths.index_select(0, perm)
        canon_valid = (
            torch.arange(opt_width, device=self.device)[None, :] < canon_lengths[:, None]
        )
        self._cached_cand_encoded = self.coe.backbone.encode_state(
            canon_opt_ids, canon_valid, causal=True
        )

    @torch.no_grad()
    def discover_batch(
        self, view: View, parent_ids: Sequence[int], start_node_id: int
    ) -> list[Outcome]:
        from .backbone import EncodedState
        from .operators import pearson_similarity

        b_cnt = len(parent_ids)
        if b_cnt == 1:
            return [self(view, parent_ids[0], start_node_id)]

        depths: list[int] = []
        scaffolds: list[str] = []
        ctx_rows: list[list[int]] = []
        extra_roots = 0
        for idx_i, pid in enumerate(parent_ids):
            d, sc, c_ids = self._prepare_node_context(
                view, pid, start_node_id + idx_i, extra_root_children=extra_roots
            )
            if pid == 0:
                extra_roots += 1
            depths.append(d)
            scaffolds.append(sc)
            ctx_rows.append(c_ids)

        pad_id = self._tok.pad_id
        max_c = max(len(r) for r in ctx_rows)
        ctx_ids = torch.tensor(
            [r + [pad_id] * (max_c - len(r)) for r in ctx_rows],
            dtype=torch.long,
            device=self.device,
        )
        ctx_lens = torch.tensor(
            [len(r) for r in ctx_rows], dtype=torch.long, device=self.device
        )
        opt_ids = self._opt_ids_1.expand(b_cnt, -1, -1)
        opt_lens = self._opt_lens_1.expand(b_cnt, -1)
        targets = self._targets_1.expand(b_cnt)

        use_amp = self.device.type == "cuda"
        op_kwargs: dict[str, Any] = {}
        if (
            int(getattr(self.coe.backbone.config, "num_experts", 4))
            * int(getattr(self.coe.backbone.config, "d_model", 128))
        ) >= 16384:
            op_kwargs = {
                "recurrence_steps": 1,
                "depth_ponder": False,
                "margin_halt_threshold": None,
            }
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            self._ensure_cand_encoded()
            base_ce = self._cached_cand_encoded
            tiled_ce = EncodedState(
                hidden=base_ce.hidden.repeat(b_cnt, 1, 1),
                valid=base_ce.valid.repeat(b_cnt, 1),
                aux_loss=base_ce.aux_loss,
                router_z_loss=base_ce.router_z_loss,
                expert_load=base_ce.expert_load,
                raw_hidden=(
                    base_ce.raw_hidden.repeat(b_cnt, 1, 1)
                    if base_ce.raw_hidden is not None
                    else None
                ),
            )
            op_out = self.coe.forward_all_operators(
                ctx_ids,
                ctx_lens,
                opt_ids,
                opt_lens,
                targets,
                candidate_encoded=tiled_ce,
                **op_kwargs,
            )
            s_feats = torch.nan_to_num(
                op_out.decision.state_features.detach().float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            q_embs = torch.nan_to_num(
                F.normalize(
                    s_feats - s_feats.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6
                ),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )

        def _clean_float(val: Any, default: float = 0.0, lo: float | None = 0.0, hi: float | None = 1.0) -> float:
            try:
                f = float(val)
            except (TypeError, ValueError, OverflowError):
                f = default
            if not math.isfinite(f):
                f = default
            if lo is not None and f < lo:
                f = lo
            if hi is not None and f > hi:
                f = hi
            return f

        probs_all = torch.nan_to_num(
            op_out.decision.probabilities.float(), nan=0.0, posinf=1.0, neginf=0.0
        )
        ded_probs_all = torch.nan_to_num(
            op_out.deduction_probs.float(), nan=0.33, posinf=1.0, neginf=0.0
        )
        meta_probs_all = torch.nan_to_num(
            op_out.metacognition_probs.float(), nan=0.5, posinf=1.0, neginf=0.0
        )
        ref_utils_all = torch.nan_to_num(
            op_out.evaluation_utilities.float().view(b_cnt, -1), nan=0.5, posinf=1.0, neginf=0.0
        )
        outcomes: list[Outcome] = []
        for idx_i, pid in enumerate(parent_ids):
            node_id = start_node_id + idx_i
            depth = depths[idx_i]
            scaffold = scaffolds[idx_i]
            probs = probs_all[idx_i]
            prob_sum = float(probs.sum().item())
            if not math.isfinite(prob_sum) or prob_sum <= 1e-8:
                probs = torch.full_like(probs, 1.0 / max(1, int(probs.numel())))
            else:
                probs = probs / prob_sum
            q_emb = q_embs[idx_i : idx_i + 1]
            expected_verdict = _clean_float((probs * self._h_scores).sum().item(), 0.0, 0.0, 1.0)
            gold_prob = _clean_float(probs[self.task.gold_index].item(), 0.0, 0.0, 1.0)
            best_rollout_idx = int(probs.argmax().item())
            picked_verdict = _clean_float(self._h_scores[best_rollout_idx].item(), 0.0, 0.0, 1.0)

            ded_entailed = _clean_float(ded_probs_all[idx_i, 0].item(), 0.33, 0.0, 1.0)
            ded_contradicted = _clean_float(ded_probs_all[idx_i, 1].item(), 0.33, 0.0, 1.0)
            meta_confidence = _clean_float(meta_probs_all[idx_i, 0].item(), 0.5, 0.0, 1.0)
            vetoed = bool((ded_contradicted > 0.55) or (meta_confidence < 0.35))
            utility = _clean_float(ref_utils_all[idx_i, best_rollout_idx].item(), 0.5, 0.0, 1.0)

            if not self._node_embeddings:
                novelty = 1.0
            else:
                prev_stack = torch.cat(self._node_embeddings, dim=0)
                sims = torch.nan_to_num(
                    pearson_similarity(q_emb.expand_as(prev_stack), prev_stack),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                novelty = _clean_float(1.0 - float(sims.max().item()), 1.0, 0.0, 2.0)
            self._node_embeddings.append(q_emb.detach())

            depth_bonus = min(0.08, 0.025 * depth) if not vetoed else -0.05
            raw_score = (
                0.65 * expected_verdict
                + 0.25 * picked_verdict
                + 0.10 * gold_prob
                + depth_bonus
            )
            score = round(_clean_float(raw_score, 0.0, 0.0, 1.0), 5)
            task_stars = _clean_float(getattr(self.task, "star_difficulty", 1.0), 1.0, 0.5, 1e6)
            task_rec = max(1, int(getattr(self.task, "recursion_depth", 1)))
            raw_pp_val = round(_clean_float(100.0 * (task_stars ** 1.15) * score, 0.0, 0.0, 1e12), 4)
            higher_in_view = sum(1 for n in view.nodes[1:] if n.score > score + 1e-7)
            pp_rank = higher_in_view + 1 + idx_i
            pp_weighted_val = round(_clean_float(raw_pp_val * (0.95 ** (pp_rank - 1)), 0.0, 0.0, 1e12), 4)
            rec_prog = round(_clean_float(min(1.0, float(depth + 1) / float(task_rec)) * expected_verdict, 0.0, 0.0, 1.0), 5)
            observation = {
                "node_id": node_id,
                "parent_id": pid,
                "depth": depth + 1,
                "scaffold": scaffold,
                "best_rollout_index": best_rollout_idx,
                "expected_verdict": round(expected_verdict, 5),
                "gold_prob": round(gold_prob, 5),
                "utility": round(utility, 5),
                "novelty": round(novelty, 5),
                "confidence": round(meta_confidence, 5),
                "deduction_entailed": round(ded_entailed, 5),
                "vetoed": vetoed,
                "star_difficulty": round(task_stars, 4),
                "recursion_depth": task_rec,
                "raw_pp": raw_pp_val,
                "pp_weighted": pp_weighted_val,
                "recursion_progress": rec_prog,
            }
            outcomes.append(Outcome(score=score, observation=observation))
        return outcomes

    @torch.no_grad()
    def __call__(self, view: View, parent_id: int, node_id: int) -> Outcome:
        from .operators import pearson_similarity

        depth, scaffold, c_ids = self._prepare_node_context(view, parent_id, node_id)
        ctx_ids = torch.tensor([c_ids], dtype=torch.long, device=self.device)
        ctx_lens = torch.tensor([len(c_ids)], dtype=torch.long, device=self.device)
        opt_ids = self._opt_ids_1
        opt_lens = self._opt_lens_1
        targets = self._targets_1

        use_amp = self.device.type == "cuda"
        op_kwargs: dict[str, Any] = {}
        if (
            int(getattr(self.coe.backbone.config, "num_experts", 4))
            * int(getattr(self.coe.backbone.config, "d_model", 128))
        ) >= 16384:
            op_kwargs = {
                "recurrence_steps": 1,
                "depth_ponder": False,
                "margin_halt_threshold": None,
            }
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            self._ensure_cand_encoded()
            op_out = self.coe.forward_all_operators(
                ctx_ids,
                ctx_lens,
                opt_ids,
                opt_lens,
                targets,
                candidate_encoded=self._cached_cand_encoded,
                **op_kwargs,
            )
            s_feat = torch.nan_to_num(
                op_out.decision.state_features[:1].detach().float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            q_emb = F.normalize(
                s_feat - s_feat.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6
            )

        def _clean_float(val: Any, default: float = 0.0, lo: float | None = 0.0, hi: float | None = 1.0) -> float:
            try:
                f = float(val)
            except (TypeError, ValueError, OverflowError):
                f = default
            if not math.isfinite(f):
                f = default
            if lo is not None and f < lo:
                f = lo
            if hi is not None and f > hi:
                f = hi
            return f

        probs = torch.nan_to_num(
            op_out.decision.probabilities[0].float(), nan=0.0, posinf=1.0, neginf=0.0
        )
        prob_sum = float(probs.sum().item())
        if not math.isfinite(prob_sum) or prob_sum <= 1e-8:
            probs = torch.full_like(probs, 1.0 / max(1, int(probs.numel())))
        else:
            probs = probs / prob_sum
        expected_verdict = _clean_float((probs * self._h_scores).sum().item(), 0.0, 0.0, 1.0)
        gold_prob = _clean_float(probs[self.task.gold_index].item(), 0.0, 0.0, 1.0)
        best_rollout_idx = int(probs.argmax().item())
        picked_verdict = _clean_float(self._h_scores[best_rollout_idx].item(), 0.0, 0.0, 1.0)

        ded_probs = torch.nan_to_num(
            op_out.deduction_probs.float(), nan=0.33, posinf=1.0, neginf=0.0
        )
        meta_probs = torch.nan_to_num(
            op_out.metacognition_probs.float(), nan=0.5, posinf=1.0, neginf=0.0
        )
        ded_entailed = _clean_float(ded_probs[0, 0].item(), 0.33, 0.0, 1.0)
        ded_contradicted = _clean_float(ded_probs[0, 1].item(), 0.33, 0.0, 1.0)
        meta_confidence = _clean_float(meta_probs[0, 0].item(), 0.5, 0.0, 1.0)
        vetoed = bool((ded_contradicted > 0.55) or (meta_confidence < 0.35))

        ref_utils = torch.nan_to_num(
            op_out.evaluation_utilities.float().view(-1), nan=0.5, posinf=1.0, neginf=0.0
        )
        utility = _clean_float(ref_utils[best_rollout_idx].item(), 0.5, 0.0, 1.0)

        if not self._node_embeddings:
            novelty = 1.0
        else:
            prev_stack = torch.cat(self._node_embeddings, dim=0)
            sims = torch.nan_to_num(
                pearson_similarity(q_emb.expand_as(prev_stack), prev_stack),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            novelty = _clean_float(1.0 - float(sims.max().item()), 1.0, 0.0, 2.0)
        self._node_embeddings.append(q_emb.detach())

        depth_bonus = min(0.08, 0.025 * depth) if not vetoed else -0.05
        raw_score = (
            0.65 * expected_verdict
            + 0.25 * picked_verdict
            + 0.10 * gold_prob
            + depth_bonus
        )
        score = round(_clean_float(raw_score, 0.0, 0.0, 1.0), 5)
        task_stars = _clean_float(getattr(self.task, "star_difficulty", 1.0), 1.0, 0.5, 1e6)
        task_rec = max(1, int(getattr(self.task, "recursion_depth", 1)))
        raw_pp_val = round(_clean_float(100.0 * (task_stars ** 1.15) * score, 0.0, 0.0, 1e12), 4)
        higher_in_view = sum(1 for n in view.nodes[1:] if n.score > score + 1e-7)
        pp_rank = higher_in_view + 1
        pp_weighted_val = round(_clean_float(raw_pp_val * (0.95 ** (pp_rank - 1)), 0.0, 0.0, 1e12), 4)
        rec_prog = round(_clean_float(min(1.0, float(depth + 1) / float(task_rec)) * expected_verdict, 0.0, 0.0, 1.0), 5)

        observation = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth + 1,
            "scaffold": scaffold,
            "best_rollout_index": best_rollout_idx,
            "expected_verdict": round(expected_verdict, 5),
            "gold_prob": round(gold_prob, 5),
            "utility": round(utility, 5),
            "novelty": round(novelty, 5),
            "confidence": round(meta_confidence, 5),
            "deduction_entailed": round(ded_entailed, 5),
            "vetoed": vetoed,
            "star_difficulty": round(task_stars, 4),
            "recursion_depth": task_rec,
            "raw_pp": raw_pp_val,
            "pp_weighted": pp_weighted_val,
            "recursion_progress": rec_prog,
        }
        return Outcome(score=score, observation=observation)


# Curated library of diverse, valid 6-line exploration policy candidates that
# `MCL4PolicyProposer` scores, mutates, and adapts using `MCL4-COE` operators + replay feedback.
_CANDIDATE_POLICY_TEMPLATES: tuple[Program, ...] = (
    Program(width=1, root_every=2, rank=("SCORE", "GAIN", "UTILITY"), stop=0.90, patience=3, rounds=4),
    Program(width=2, root_every=2, rank=("SCORE", "GAIN", "UTILITY"), stop=0.88, patience=3, rounds=4),
    Program(width=2, root_every=2, rank=("PP", "UTILITY", "SCORE", "GAIN"), stop=0.87, patience=2, rounds=3),
    Program(width=2, root_every=2, rank=("RECURSION", "PP", "UTILITY", "SCORE"), stop=0.86, patience=2, rounds=4),
    Program(width=2, root_every=2, rank=("UTILITY", "SCORE", "GAIN"), stop=0.85, patience=2, rounds=3),
    Program(width=2, root_every=1, rank=("SCORE", "GAIN", "UTILITY"), stop=0.84, patience=2, rounds=3),
    Program(width=2, root_every=2, rank=("UTILITY", "CONFIDENCE", "SCORE", "GAIN"), stop=0.82, patience=2, rounds=3),
    Program(width=2, root_every=2, rank=("CONFIDENCE", "UTILITY", "SCORE", "DEEP"), stop=0.80, patience=2, rounds=3),
    Program(width=2, root_every=2, rank=("GAIN", "UTILITY", "SCORE", "NOVELTY"), stop=0.78, patience=2, rounds=3),
    Program(width=2, root_every=2, rank=("SCORE", "UTILITY", "CONFIDENCE", "GAIN"), stop=0.76, patience=1, rounds=2),
    Program(width=2, root_every=0, rank=("SCORE", "DEEP", "UTILITY", "GAIN"), stop=0.75, patience=1, rounds=2),
    Program(width=2, root_every=2, rank=("UTILITY", "SCORE", "GAIN", "CONFIDENCE"), stop=0.73, patience=1, rounds=2),
    Program(width=2, root_every=2, rank=("CONFIDENCE", "SCORE", "GAIN", "UTILITY"), stop=0.71, patience=1, rounds=2),
    Program(width=2, root_every=1, rank=("UTILITY", "SCORE", "GAIN", "CONFIDENCE"), stop=None, patience=2, rounds=3),
)

_MUTATION_RANK_ORDERS: tuple[tuple[str, ...], ...] = (
    ("SCORE", "UTILITY", "CONFIDENCE", "GAIN"),
    ("PP", "UTILITY", "SCORE", "GAIN"),
    ("RECURSION", "PP", "UTILITY", "CONFIDENCE"),
    ("UTILITY", "CONFIDENCE", "SCORE", "NOVELTY"),
    ("CONFIDENCE", "UTILITY", "GAIN", "SCORE"),
    ("GAIN", "SCORE", "UTILITY", "DEEP"),
    ("NOVELTY", "UTILITY", "SCORE", "GAIN"),
    ("SCORE", "GAIN", "CONFIDENCE", "UTILITY"),
)


class MCL4PolicyProposer:
    """Proposes and mutates candidate 6-line `Program` revisions conditioned on replay diagnostics.

    Uses `MCL4-COE`'s `CognitiveOperatorEnsemble` (`Decision` + `Evaluation` +
    `Metacognition`) when a live model is attached to rank synthesized policy revisions
    against the replay feedback summary, and returns `{"text": str, "finished": True}`.
    """

    def __init__(self, coe: Any | None = None, *, seed: int = 42) -> None:
        self.coe = coe
        self.seed = seed
        self._call_count = 0

    def _synthesize_mutated_programs(
        self,
        incumbent: Program,
        *,
        mean_hv: float,
        mean_penalty: float,
        max_w: int,
        max_r: int,
        revision: int,
    ) -> list[Program]:
        """Synthesize mutated and crossover 6-line `Program` policies around `incumbent`."""
        mutated: list[Program] = []
        base_stop = 0.84 if incumbent.stop is None else float(incumbent.stop)
        rank_choice = _MUTATION_RANK_ORDERS[(self.seed + revision + self._call_count) % len(_MUTATION_RANK_ORDERS)]

        # Mutation 1: Feature-rank crossover with adaptive stop threshold tightening
        stop_m1 = round(max(0.68, min(0.94, base_stop + (-0.02 if mean_hv < 0.65 else 0.02))), 2)
        mutated.append(
            Program(
                width=min(max_w, max(2, incumbent.width)),
                root_every=2 if incumbent.root_every != 2 else 1,
                rank=rank_choice,
                stop=stop_m1,
                patience=max(1, min(3, (incumbent.patience or 1) + (1 if mean_hv < 0.55 else 0))),
                rounds=min(max_r, max(2, incumbent.rounds)),
            )
        )

        # Mutation 2: Depth-focused leaf refinement (root_every=2 or 0) when penalty is high
        stop_m2 = round(max(0.68, min(0.92, base_stop - 0.03)), 2)
        mutated.append(
            Program(
                width=min(max_w, 2),
                root_every=2 if mean_penalty <= 0.40 else 0,
                rank=("UTILITY", "CONFIDENCE", "SCORE", "GAIN"),
                stop=stop_m2,
                patience=1 if mean_penalty > 0.30 else 2,
                rounds=min(max_r, max(2, incumbent.rounds - (1 if mean_penalty > 0.35 else 0))),
            )
        )
        return mutated

    @torch.no_grad()
    def __call__(
        self, incumbent: Program, feedback: dict[str, Any], revision: int
    ) -> dict[str, Any]:
        self._call_count += 1
        world_summary = feedback.get("world_summary", {})
        mean_penalty = float(world_summary.get("mean_parallel_penalty", 0.25))
        mean_hv = float(world_summary.get("mean_hypervolume", 0.3))
        budgets = feedback.get("budgets", {})
        max_w = max(1, int(budgets.get("workers", 4)))
        max_r = max(1, int(budgets.get("max_rounds", 8)))

        # Construct adaptive candidate programs informed by replay diagnostics + genetic mutation
        dynamic_candidates: list[Program] = []
        if incumbent.width < max_w or incumbent.root_every == 1:
            dynamic_candidates.append(
                Program(
                    width=min(max_w, max(1, incumbent.width)),
                    root_every=2,
                    rank=("SCORE", "GAIN", "UTILITY"),
                    stop=0.90,
                    patience=3,
                    rounds=min(max_r, incumbent.rounds),
                )
            )
        if incumbent.rounds > 2:
            dynamic_candidates.append(
                Program(
                    width=min(max_w, max(2, incumbent.width)),
                    root_every=2,
                    rank=("SCORE", "GAIN", "UTILITY"),
                    stop=0.82 if incumbent.stop is None else max(0.70, round(float(incumbent.stop) - 0.03, 2)),
                    patience=max(1, min(2, incumbent.patience or 2)),
                    rounds=min(max_r, max(2, incumbent.rounds - 1)),
                )
            )
        if mean_penalty > 0.35:
            dynamic_candidates.append(
                Program(
                    width=min(max_w, max(2, incumbent.width)),
                    root_every=2,
                    rank=("UTILITY", "SCORE", "GAIN", "CONFIDENCE"),
                    stop=0.76 if mean_hv < 0.55 else 0.80,
                    patience=1,
                    rounds=min(max_r, max(2, incumbent.rounds - 1)),
                )
            )
        if mean_hv < 0.65:
            stop_cand = 0.78 if incumbent.stop is None else max(0.68, round(float(incumbent.stop) - 0.02, 2))
            dynamic_candidates.append(
                Program(
                    width=min(max_w, 2),
                    root_every=2,
                    rank=("UTILITY", "CONFIDENCE", "SCORE", "GAIN"),
                    stop=stop_cand,
                    patience=1,
                    rounds=min(max_r, 2),
                )
            )
        for mut_p in self._synthesize_mutated_programs(
            incumbent,
            mean_hv=mean_hv,
            mean_penalty=mean_penalty,
            max_w=max_w,
            max_r=max_r,
            revision=revision,
        ):
            if mut_p != incumbent and mut_p not in dynamic_candidates:
                dynamic_candidates.append(mut_p)

        for tpl in _CANDIDATE_POLICY_TEMPLATES:
            adapted = Program(
                width=min(max_w, tpl.width),
                root_every=tpl.root_every,
                rank=tpl.rank,
                stop=tpl.stop,
                patience=tpl.patience,
                rounds=min(max_r, tpl.rounds),
            )
            if adapted != incumbent and adapted not in dynamic_candidates:
                dynamic_candidates.append(adapted)

        if not dynamic_candidates:
            dynamic_candidates.append(incumbent)

        if self.coe is not None and len(dynamic_candidates) >= 4:
            from .data import get_tokenizer

            dev = next(self.coe.parameters()).device
            offset = (revision + self._call_count - 2) % len(dynamic_candidates)
            rotated = dynamic_candidates[offset:] + dynamic_candidates[:offset]
            four_cands = rotated[:4]
            fb_prompt = (
                f"#U$# Dream-RSI Policy Revision {revision}: "
                f"score={feedback.get('current_score', 0.0):.4f}, "
                f"hv={mean_hv:.4f}, penalty={mean_penalty:.4f}\n"
                f"#Q$# Select the 6-line exploration policy that maximizes Pareto hypervolume."
            )
            vocab_size = int(getattr(self.coe.backbone.config, "vocab_size", 260))
            tok = get_tokenizer(vocab_size)
            pad_id = tok.pad_id
            ctx_toks = tok.encode(fb_prompt, add_bos=True, add_eos=True)[:256]
            cand_rows = [
                tok.encode(c.to_text(), add_bos=True, add_eos=True)[:128]
                for c in four_cands
            ]
            max_o = max(len(r) for r in cand_rows)
            ctx_ids = torch.tensor([ctx_toks], dtype=torch.long, device=dev)
            ctx_valid = torch.ones_like(ctx_ids, dtype=torch.bool)
            opt_ids = torch.tensor(
                [r + [pad_id] * (max_o - len(r)) for r in cand_rows],
                dtype=torch.long,
                device=dev,
            )
            opt_valid = torch.tensor(
                [[i < len(r) for i in range(max_o)] for r in cand_rows],
                dtype=torch.bool,
                device=dev,
            )
            owner_rows = torch.zeros(4, dtype=torch.long, device=dev)
            use_amp = dev.type == "cuda"
            with torch.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                emb_mult = float(
                    getattr(self.coe.backbone.config, "embedding_multiplier", 1.0)
                )
                ctx_h = (
                    self.coe.backbone.encoder_norm(
                        self.coe.backbone.token_embedding(ctx_ids) * emb_mult
                    )
                    * ctx_valid.unsqueeze(-1)
                )
                opt_h = (
                    self.coe.backbone.final_norm(
                        self.coe.backbone.token_embedding(opt_ids) * emb_mult
                    )
                    * opt_valid.unsqueeze(-1)
                )
                paired_state_repr, cand_repr = (
                    self.coe.pooler.matched_pair_representations(
                        ctx_h,
                        ctx_ids,
                        ctx_valid,
                        opt_h,
                        opt_ids,
                        opt_valid,
                        owner_rows=owner_rows,
                    )
                )
                from .operators import pearson_similarity

                s_proj = self.coe.state_projection(paired_state_repr)
                c_proj = self.coe.output_projection(cand_repr)
                sims = self.coe._similarity(s_proj, c_proj)
                scale = self.coe._scale()
                d_probs = (sims.float() * scale).softmax(dim=-1)
                e_utils = (
                    (sims.float() * self.coe.assoc_alpha.float() + self.coe.assoc_beta.float())
                    .sigmoid()
                    .view(-1)
                    .clamp(0.0, 1.0)
                )
                meta_joint = self.coe.meta_query_proj(
                    torch.cat([paired_state_repr, cand_repr], dim=-1)
                )
                meta_sims = pearson_similarity(
                    meta_joint.unsqueeze(1),
                    self.coe.meta_prototypes.unsqueeze(0),
                    eps=self.coe.config.eps,
                )
                m_conf = (
                    (meta_sims.float() * scale + self.coe.meta_bias.float().unsqueeze(0))
                    .sigmoid()[:, 0]
                    .view(-1)
                    .clamp(0.0, 1.0)
                )
            joint_score = 0.60 * d_probs + 0.25 * e_utils + 0.15 * m_conf
            pick = int(joint_score.argmax().item())
            chosen = four_cands[pick]
        else:
            chosen = dynamic_candidates[(revision - 1) % len(dynamic_candidates)]

        return {
            "text": chosen.to_text(),
            "finished": True,
            "within_byte_budget": True,
            "sampling_calls": 1,
        }


@dataclass(frozen=True)
class DreamCycleOutput:
    """Telemetry from one complete Online-Explore + Offline-Dream cycle."""

    cycle: int
    world_id: str
    best_node_score: float
    best_scaffold: str
    online_calls: int
    online_rounds: int
    initial_replay_score: float
    final_replay_score: float
    accepted_revision: int
    total_accepted_revisions: int
    replays_dreamed: int
    total_replays_dreamed: int
    worlds_archived: int
    active_policy_text: str
    active_policy: dict[str, Any]


class DreamRSIEngine:
    """Executes the recursive Dream-RSI loop inside `MCL4-COE`:

    $$\\mathcal{H}_t = (\\mathcal{T}_1, \\dots, \\mathcal{T}_t)
      \\;\\xrightarrow{\\;\\text{replay}\\;}\\;
      \\pi_{t+1} \\;\\xrightarrow{\\;\\text{deploy}\\;}\\; \\mathcal{T}_{t+1}$$
    """

    def __init__(
        self,
        *,
        initial_program: Program | None = None,
        max_rounds: int = 4,
        workers: int = 2,
        max_calls: int = 6,
        revisions_per_cycle: int = 3,
        round_penalty: float = 0.05,
        rhos: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        max_worlds: int = 48,
        out_dir: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        default_prog = initial_program or Program(
            width=1,
            root_every=1,
            rank=("SHALLOW", "SCORE"),
            stop=None,
            patience=0,
            rounds=max_rounds,
        )
        self.state = DreamState(program=default_prog, initial_program=default_prog)
        self.max_rounds = max_rounds
        self.workers = workers
        self.max_calls = max_calls
        self.revisions_per_cycle = revisions_per_cycle
        self.round_penalty = round_penalty
        self.rhos = tuple(rhos)
        self.max_worlds = max_worlds
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.seed = seed

        if self.out_dir is not None:
            state_file = self.out_dir / "dream_state.json"
            if state_file.exists():
                try:
                    raw = json.loads(state_file.read_text(encoding="utf-8"))
                    self.state = DreamState.from_dict(raw)
                except Exception:
                    pass

    def save_artifacts(self, improvement_receipt: dict[str, Any] | None = None) -> None:
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.out_dir / "dream_state.json", self.state.to_dict())
        _atomic_write_json(
            self.out_dir / "dream_policy.json",
            {
                "cycle": self.state.cycle,
                "program": self.state.program.to_dict(),
                "program_text": self.state.program.to_text(),
                "initial_score": self.state.initial_score,
                "latest_score": self.state.latest_score,
                "accepted_revisions": self.state.accepted_revisions,
                "total_replays_dreamed": self.state.total_replays_dreamed,
            },
        )
        if improvement_receipt is not None:
            rev_path = self.out_dir / "dream_revisions.jsonl"
            line = json.dumps(
                {
                    "cycle": self.state.cycle,
                    "initial_score": improvement_receipt["initial_score"],
                    "final_score": improvement_receipt["final_score"],
                    "accepted_revision": improvement_receipt["accepted_revision"],
                    "replays_dreamed": improvement_receipt["replays_dreamed"],
                    "program_text": improvement_receipt["program_text"],
                },
                sort_keys=True,
            )
            with rev_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def step_cycle(
        self,
        coe: Any,
        task: Any,
        *,
        rho_online: float = 1.0,
        max_seq_len: int = 384,
    ) -> DreamCycleOutput:
        """Run one complete Dream-RSI iteration: Online Explore ($\\mathcal{T}_t$) + Offline Dream ($\\pi_{t+1}$)."""
        self.state.cycle += 1
        cycle_idx = self.state.cycle
        world_id = f"world_{cycle_idx:04d}_{task.task_id}"

        discovery = MCL4DreamDiscovery(coe, task, max_seq_len=max_seq_len)
        # Ensure online tree expansion uses the active policy's feature ranking while exploring
        # both root branches and depth-2+ leaf refinements up to `max_calls` so the recorded
        # `World` supports rich counterfactual replay across stopping and width strategies.
        online_explorer = Program(
            width=max(self.workers, self.state.program.width),
            root_every=2 if (self.state.program.width <= 1 and self.state.program.root_every == 1) else max(2, self.state.program.root_every),
            rank=self.state.program.rank,
            stop=None,
            patience=0,
            rounds=max(self.max_rounds, self.state.program.rounds),
        )
        task_skill = str(getattr(task, "skill", "general.reasoning"))
        task_stars = float(getattr(task, "star_difficulty", 1.0))
        task_rec_depth = int(getattr(task, "recursion_depth", 1))
        online_res = online_rollout(
            online_explorer,
            discovery,
            world_id=world_id,
            root_observation={
                "task_id": task.task_id,
                "domain": task.domain,
                "skill": task_skill,
                "star_difficulty": round(task_stars, 4),
                "recursion_depth": task_rec_depth,
            },
            metadata={
                "cycle": cycle_idx,
                "domain": task.domain,
                "skill": task_skill,
                "star_difficulty": round(task_stars, 4),
                "recursion_depth": task_rec_depth,
            },
            max_rounds=self.max_rounds,
            workers=self.workers,
            max_calls=self.max_calls,
            rho=rho_online,
        )

        self.state.total_calls += online_res.agent_calls
        self.state.append_world(online_res.world, max_worlds=self.max_worlds)

        # Find best discovered node and its refined #S$# cognitive operator scaffold
        non_root = online_res.world.nodes[1:]
        if non_root:
            best_node = max(non_root, key=lambda n: n.score)
            best_score = float(best_node.score)
            best_scaffold = str(
                best_node.observation.get("scaffold", task.scaffold_s)
            )
        else:
            best_score = 0.0
            best_scaffold = task.scaffold_s

        # Offline Dreaming: Propose candidate policies and score over H_t at ZERO execution cost
        proposer = MCL4PolicyProposer(coe, seed=self.seed + cycle_idx)
        imp = improve_policy(
            self.state.worlds,
            self.state.program,
            proposer,
            revisions=self.revisions_per_cycle,
            max_rounds=self.max_rounds,
            workers=self.workers,
            max_calls=self.max_calls,
            round_penalty=self.round_penalty,
            rhos=self.rhos,
            revision_offset=self.state.total_revisions,
        )

        self.state.program = Program.from_dict(imp["program"])
        self.state.total_revisions += self.revisions_per_cycle
        if imp["accepted_revision"] > 0:
            self.state.accepted_revisions += 1
        self.state.total_replays_dreamed += int(imp["replays_dreamed"])
        if cycle_idx == 1:
            self.state.initial_score = float(imp["initial_score"])
        self.state.latest_score = float(imp["final_score"])

        self.state.add_receipt(
            {
                "cycle": cycle_idx,
                "world_id": world_id,
                "online_calls": online_res.agent_calls,
                "online_rounds": online_res.rounds,
                "best_node_score": best_score,
                "initial_replay_score": round(float(imp["initial_score"]), 6),
                "final_replay_score": round(float(imp["final_score"]), 6),
                "accepted_revision": imp["accepted_revision"],
                "active_policy": self.state.program.to_text(),
            }
        )
        if cycle_idx <= 4 or cycle_idx % 20 == 0 or imp["accepted_revision"] > 0:
            self.save_artifacts(imp)

        return DreamCycleOutput(
            cycle=cycle_idx,
            world_id=world_id,
            best_node_score=round(best_score, 4),
            best_scaffold=best_scaffold,
            online_calls=online_res.agent_calls,
            online_rounds=online_res.rounds,
            initial_replay_score=round(float(imp["initial_score"]), 5),
            final_replay_score=round(float(imp["final_score"]), 5),
            accepted_revision=int(imp["accepted_revision"]),
            total_accepted_revisions=self.state.accepted_revisions,
            replays_dreamed=int(imp["replays_dreamed"]),
            total_replays_dreamed=self.state.total_replays_dreamed,
            worlds_archived=len(self.state.worlds),
            active_policy_text=self.state.program.to_text(),
            active_policy=self.state.program.to_dict(),
        )

    @torch.no_grad()
    def run_user_context_dream_session(
        self,
        coe: Any,
        user_buffer: "UserContextDreamBuffer",
        dream_cache: "DreamContextCache",
        *,
        max_cycles: int | None = None,
        rollout_tokens: int = 160,
        seed: int = 42,
    ) -> list[DreamCycleOutput]:
        """Run idle or manual Dream-RSI cycles seeded strictly with human `#U$#` contexts.

        Leaves out all model-generated prompts/responses in the seed buffer, evolves the
        Dream-RSI exploration policy $\\pi_t$, and distills a zero-token latent `Abstraction`
        prior ($\\mathbf{z}_{\\text{abs}}^{*} \\in \\mathbb{R}^{d_{\\text{model}}}$) + optimal `#S$#`
        cognitive operator scaffold into `dream_cache` for future similar or adjacent OOD topics.
        """
        from .data import get_tokenizer
        from .ensemble import MultiPassController, _score_trace_integrity
        from .ornith_rl import OrnithTask

        was_training = coe.training
        coe.eval()
        outputs: list[DreamCycleOutput] = []
        try:
            device = next(coe.parameters()).device
            vocab_size = int(getattr(coe.backbone.config, "vocab_size", 260))
            tokenizer = get_tokenizer(vocab_size)
            controller = MultiPassController(coe)
            records = user_buffer.pending_records(limit=max_cycles)

            for idx, rec in enumerate(records):
                u_prompt = rec.user_context
                # Generate 4 candidate trajectories from the human #U$# context to evaluate scaffolds
                cand_specs = ((1, 0.0), (2, 0.0), (1, 0.28), (2, 0.35))
                cand_bodies: list[str] = []
                for c_idx, (r_steps, temp) in enumerate(cand_specs):
                    if temp > 0.0:
                        torch.manual_seed(seed + idx * 97 + c_idx * 31)
                    tr = controller.generate_uta(
                        u_prompt,
                        max_new_tokens=rollout_tokens,
                        temperature=temp,
                        recurrence_steps=r_steps,
                    )
                    t_pos = tr.find("#T$#")
                    body = tr[t_pos:].strip() if t_pos >= 0 else tr.strip()
                    cand_bodies.append(body if body else "#T$# empty\n#A$# empty")

                # Compute self-supervised structural + COE harness scores for the 4 rollouts
                h_scores = tuple(
                    round(max(0.05, _score_trace_integrity(u_prompt, f"#U$# {u_prompt}\n{b}")), 4)
                    for b in cand_bodies
                )
                gold_idx = int(max(range(len(h_scores)), key=lambda i: h_scores[i]))
                best_body = cand_bodies[gold_idx]
                if "#A$#" in best_body:
                    t_seg, a_seg = best_body.split("#A$#", 1)
                    gold_t = t_seg.replace("#T$#", "", 1).strip()
                    gold_a = a_seg.strip()
                else:
                    gold_t = best_body.replace("#T$#", "", 1).strip()
                    gold_a = best_body

                task = OrnithTask(
                    task_id=rec.context_id,
                    domain=rec.domain,
                    prompt_q=f"#U$# {u_prompt}",
                    harness_h="user_context_dream_rsi",
                    scaffold_s="abduction -> association -> decision -> deduction -> evaluation -> metacognition",
                    candidate_rollouts=tuple(cand_bodies),
                    harness_scores=h_scores,
                    gold_index=gold_idx,
                    gold_thought=gold_t,
                    gold_answer=gold_a,
                )

                cycle_out = self.step_cycle(
                    coe,
                    task,
                    max_seq_len=int(getattr(coe.backbone.config, "max_seq_len", 512)),
                )
                rec.dreamed_cycles += 1
                outputs.append(cycle_out)

                # Distill zero-token latent Abstraction prior z_abs* and 32-d Pearson intent key q_emb
                max_seq = int(getattr(coe.backbone.config, "max_seq_len", 512))
                u_ids = tokenizer.encode(f"#U$# {u_prompt}", add_bos=True, add_eos=True)[:max_seq]
                sc_ids = tokenizer.encode(
                    f"#U$# {u_prompt}\n#S$# {cycle_out.best_scaffold}",
                    add_bos=True,
                    add_eos=True,
                )[:max_seq]
                b_ids = tokenizer.encode(best_body, add_bos=True, add_eos=True)[:max_seq]

                u_t = torch.tensor([u_ids], dtype=torch.long, device=device)
                u_len = torch.tensor([len(u_ids)], dtype=torch.long, device=device)
                sc_t = torch.tensor([sc_ids], dtype=torch.long, device=device)
                sc_len = torch.tensor([len(sc_ids)], dtype=torch.long, device=device)
                b_t = torch.tensor([b_ids], dtype=torch.long, device=device)
                b_len = torch.tensor([len(b_ids)], dtype=torch.long, device=device)

                use_amp = device.type == "cuda"
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    q_emb = coe.embed_items(u_t, u_len)[0].detach().float().cpu()
                    z_prompt = coe.compress_abstraction(sc_t, sc_len, level="standard")[0].float()
                    z_cand = coe.compress_abstraction(b_t, b_len, level="detailed")[0].float()
                    z_blend = coe.abstraction_norm(
                        (0.65 * z_prompt + 0.35 * z_cand).to(
                            next(coe.abstraction_norm.parameters()).dtype
                        )
                    ).detach().float().cpu()

                dream_cache.upsert(
                    DreamContextEntry(
                        context_id=rec.context_id,
                        user_context=u_prompt,
                        domain=rec.domain,
                        q_emb=tuple(round(float(x), 6) for x in q_emb.tolist()),
                        z_abs=tuple(round(float(x), 6) for x in z_blend.tolist()),
                        best_scaffold=cycle_out.best_scaffold,
                        best_node_score=cycle_out.best_node_score,
                        active_policy_text=cycle_out.active_policy_text,
                    )
                )

            if self.out_dir is not None:
                dream_cache.save(self.out_dir / "dream_context_cache.json")
            return outputs
        finally:
            coe.train(was_training)


_MODEL_GENERATED_CUT_MARKERS: tuple[str, ...] = (
    "#T$#",
    "#A$#",
    "#S$#",
    "#H$#",
    "#Q$#",
    "#H_ANC$#",
    "<eos>",
    "\nAssistant:",
    "\nModel:",
    "\nAI:",
)

_SYNTHETIC_SELF_PROMPT_PREFIXES: tuple[str, ...] = (
    "the user asks:",
    "user asks:",
    "let's analyze",
    "we need to",
    "step 1:",
)


def extract_user_only_context(raw_text: str) -> str:
    """Strictly extract human-authored `#U$#` context, stripping all model-generated prompts/outputs.

    Prevents autophagous mode collapse during idle/manual Dream-RSI by guaranteeing that
    no `#T$#` scratchpads, `#A$#` answers, `#S$#` scaffolds, or synthetic self-prompts enter
    the Dream seed buffer.
    """
    text = str(raw_text or "").replace("<bos>", "").strip()
    if not text:
        return ""

    # If explicit #U$# markers are present (e.g., multi-turn transcript), extract only human #U$# segments
    if "#U$#" in text:
        parts = text.split("#U$#")[1:]
        user_turns: list[str] = []
        for part in parts:
            seg = part
            for marker in _MODEL_GENERATED_CUT_MARKERS:
                idx = seg.find(marker)
                if idx >= 0:
                    seg = seg[:idx]
            cleaned = " ".join(seg.strip().split())
            if cleaned and not cleaned.lower().startswith(_SYNTHETIC_SELF_PROMPT_PREFIXES):
                user_turns.append(cleaned)
        return " ".join(user_turns).strip()

    # Otherwise strip from the earliest model-generated marker onward
    seg = text
    for marker in _MODEL_GENERATED_CUT_MARKERS:
        idx = seg.find(marker)
        if idx >= 0:
            seg = seg[:idx]
    cleaned = " ".join(seg.strip().split())
    if cleaned.lower().startswith(_SYNTHETIC_SELF_PROMPT_PREFIXES):
        return ""
    return cleaned


@dataclass
class UserContextRecord:
    """A sanitized human-only user prompt/context record for seeding idle/manual Dream-RSI."""

    context_id: str
    user_context: str
    domain: str = "general"
    dreamed_cycles: int = 0


class UserContextDreamBuffer:
    """Buffer of strictly human-provided `#U$#` contexts for idle or manual Dream-RSI seeding."""

    def __init__(self, max_size: int = 256) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._records: list[UserContextRecord] = []
        self._seen_texts: set[str] = set()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[UserContextRecord, ...]:
        return tuple(self._records)

    def add_user_context(
        self,
        raw_text: str,
        *,
        domain: str = "general",
        context_id: str | None = None,
    ) -> UserContextRecord | None:
        """Sanitize `raw_text` (stripping any model `#T$#`/`#A$#` text) and append if non-empty & unique."""
        clean = extract_user_only_context(raw_text)
        if len(clean) < 8:
            return None
        key = clean.lower()
        if key in self._seen_texts:
            for existing in self._records:
                if existing.user_context.lower() == key:
                    return existing
            return None
        cid = context_id or f"uctx_{len(self._records) + 1:04d}"
        rec = UserContextRecord(context_id=cid, user_context=clean, domain=domain)
        if len(self._records) >= self.max_size:
            evicted = self._records.pop(0)
            self._seen_texts.discard(evicted.user_context.lower())
        self._records.append(rec)
        self._seen_texts.add(key)
        return rec

    def add_conversation_history(
        self,
        turns: Sequence[str | dict[str, Any]],
        *,
        domain: str = "general",
    ) -> int:
        """Ingest a conversation history, keeping ONLY user turns and discarding all model turns."""
        added = 0
        for item in turns:
            if isinstance(item, dict):
                role = str(item.get("role", "user")).lower()
                if role not in ("user", "human"):
                    continue
                content = str(item.get("content", ""))
                dom = str(item.get("domain", domain))
            else:
                content = str(item)
                dom = domain
            if self.add_user_context(content, domain=dom) is not None:
                added += 1
        return added

    def pending_records(self, limit: int | None = None) -> list[UserContextRecord]:
        """Return user contexts ordered by fewest dreamed cycles first."""
        ordered = sorted(self._records, key=lambda r: r.dreamed_cycles)
        return ordered if limit is None else ordered[: max(0, int(limit))]


@dataclass(frozen=True)
class DreamContextEntry:
    """Distilled zero-token latent prior from a Dream-RSI cycle on a user context."""

    context_id: str
    user_context: str
    domain: str
    q_emb: tuple[float, ...]
    z_abs: tuple[float, ...]
    best_scaffold: str
    best_node_score: float
    active_policy_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "user_context": self.user_context,
            "domain": self.domain,
            "q_emb": list(self.q_emb),
            "z_abs": list(self.z_abs),
            "best_scaffold": self.best_scaffold,
            "best_node_score": self.best_node_score,
            "active_policy_text": self.active_policy_text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DreamContextEntry":
        return cls(
            context_id=str(data["context_id"]),
            user_context=str(data["user_context"]),
            domain=str(data.get("domain", "general")),
            q_emb=tuple(float(x) for x in data["q_emb"]),
            z_abs=tuple(float(x) for x in data["z_abs"]),
            best_scaffold=str(data["best_scaffold"]),
            best_node_score=float(data["best_node_score"]),
            active_policy_text=str(data.get("active_policy_text", "")),
        )


@dataclass(frozen=True)
class DreamRetrievalResult:
    """Zero-token prior retrieved from `DreamContextCache` for a new or adjacent OOD user prompt."""

    has_match: bool
    top_similarity: float
    matched_contexts: tuple[str, ...]
    matched_domains: tuple[str, ...]
    matched_similarities: tuple[float, ...]
    interpolated_z_abs: torch.Tensor | None
    recommended_scaffold: str
    recommended_policy_text: str


class DreamContextCache:
    """Latent episodic cache mapping user intent embeddings to pre-dreamed `#S$#` scaffolds & `Abstraction` vectors."""

    def __init__(self) -> None:
        self._entries: dict[str, DreamContextEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[DreamContextEntry, ...]:
        return tuple(self._entries.values())

    def upsert(self, entry: DreamContextEntry) -> None:
        self._entries[entry.context_id] = entry

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [e.to_dict() for e in self._entries.values()]}
        _atomic_write_json(p, payload)

    @classmethod
    def load(cls, path: str | Path) -> "DreamContextCache":
        cache = cls()
        p = Path(path)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            for item in raw.get("entries", []):
                cache.upsert(DreamContextEntry.from_dict(item))
        return cache

    @staticmethod
    def _extract_morpheme_features(text: str) -> set[str]:
        import re

        stop = {
            "what", "when", "where", "which", "that", "this", "with", "from",
            "into", "your", "does", "have", "explain", "how", "why", "write",
            "find", "finds", "between", "even", "less", "than", "each", "more",
        }
        words = [
            w
            for w in re.findall(r"[a-z0-9_]{3,}", text.lower())
            if w not in stop
        ]
        feats: set[str] = set()
        for w in words:
            feats.add(f"w:{w[:6]}")
            if len(w) >= 4:
                for i in range(len(w) - 3):
                    feats.add(f"g:{w[i:i+4]}")
        return feats

    @torch.no_grad()
    def retrieve(
        self,
        query_q_emb: torch.Tensor,
        *,
        query_z_abs: torch.Tensor | None = None,
        query_text: str | None = None,
        whiten: bool = True,
        top_k: int = 3,
        min_similarity: float = 0.05,
        exclude_exact_text: str | None = None,
        device: str | torch.device | None = None,
    ) -> DreamRetrievalResult:
        """Retrieve top-$k$ adjacent user-context priors in 0 prompt tokens via whitened Pearson similarity."""
        from .operators import pearson_similarity

        default_scaffold = "abduction -> association -> decision -> deduction -> evaluation -> metacognition"
        if not self._entries:
            return DreamRetrievalResult(
                has_match=False,
                top_similarity=0.0,
                matched_contexts=(),
                matched_domains=(),
                matched_similarities=(),
                interpolated_z_abs=None,
                recommended_scaffold=default_scaffold,
                recommended_policy_text="",
            )

        excl = extract_user_only_context(exclude_exact_text).lower() if exclude_exact_text else None
        candidates = [
            e
            for e in self._entries.values()
            if excl is None or e.user_context.lower() != excl
        ]
        if not candidates:
            return DreamRetrievalResult(
                has_match=False,
                top_similarity=0.0,
                matched_contexts=(),
                matched_domains=(),
                matched_similarities=(),
                interpolated_z_abs=None,
                recommended_scaffold=default_scaffold,
                recommended_policy_text="",
            )

        target_dev = (
            torch.device(device)
            if device is not None
            else query_q_emb.device
        )
        q_vec = query_q_emb.detach().float().view(1, -1).to(target_dev)
        keys_q = torch.tensor(
            [e.q_emb for e in candidates], dtype=torch.float32, device=target_dev
        )

        # 1. Cache-Key Whitening on 32-d q_emb: removes the shared #U$# prompt-cone mean
        if whiten and len(candidates) >= 2:
            mu_q = keys_q.mean(dim=0, keepdim=True)
            std_q = keys_q.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            q_whitened = (q_vec - mu_q) / std_q
            keys_q_whitened = (keys_q - mu_q) / std_q
            sims_q = pearson_similarity(q_whitened.expand_as(keys_q_whitened), keys_q_whitened).view(-1)
        else:
            sims_q = pearson_similarity(q_vec.expand_as(keys_q), keys_q).view(-1)

        # 2. Optional 384-d Abstraction Space Whitening (query_z_abs vs cached z_abs)
        z_stack_all = torch.tensor(
            [e.z_abs for e in candidates], dtype=torch.float32, device=target_dev
        )
        if query_z_abs is not None and query_z_abs.numel() == z_stack_all.shape[-1]:
            qz = query_z_abs.detach().float().view(1, -1).to(target_dev)
            if whiten and len(candidates) >= 2:
                mu_z = z_stack_all.mean(dim=0, keepdim=True)
                std_z = z_stack_all.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
                qz_w = (qz - mu_z) / std_z
                kz_w = (z_stack_all - mu_z) / std_z
                sims_z = pearson_similarity(qz_w.expand_as(kz_w), kz_w).view(-1)
            else:
                sims_z = pearson_similarity(qz.expand_as(z_stack_all), z_stack_all).view(-1)
        else:
            sims_z = sims_q

        # 3. IDF-weighted Morpheme / Stem similarity for sharp domain disambiguation
        q_clean = extract_user_only_context(query_text or exclude_exact_text or "")
        if q_clean:
            q_feats = self._extract_morpheme_features(q_clean)
            cand_feats = [self._extract_morpheme_features(e.user_context) for e in candidates]
            df: dict[str, int] = {}
            for cf in cand_feats:
                for f in cf:
                    df[f] = df.get(f, 0) + 1
            n_docs = len(candidates)
            lex_scores: list[float] = []
            for cf in cand_feats:
                if not q_feats or not cf:
                    lex_scores.append(0.0)
                    continue
                inter = q_feats & cf
                if not inter:
                    lex_scores.append(0.0)
                    continue
                w_inter = sum(math.log(1.0 + n_docs / max(1, df.get(f, 1))) for f in inter)
                w_q = sum(math.log(1.0 + n_docs / max(1, df.get(f, 1))) for f in q_feats)
                w_c = sum(math.log(1.0 + n_docs / max(1, df.get(f, 1))) for f in cf)
                lex_scores.append(w_inter / math.sqrt(max(1e-6, w_q * w_c)))
            sims_lex_raw = torch.tensor(lex_scores, dtype=torch.float32, device=target_dev)
            sims_lex = (sims_lex_raw / sims_lex_raw.max().clamp_min(0.20)).clamp(0.0, 1.0)
            # Combine whitened neural similarity (mapped from [-1, 1] to [0, 1]) with calibrated morphemic similarity
            neural_01 = (0.55 * sims_q + 0.45 * sims_z).clamp(-1.0, 1.0) * 0.5 + 0.5
            sims = 0.50 * neural_01 + 0.50 * sims_lex
        else:
            sims = (0.60 * sims_q + 0.40 * sims_z).clamp(-1.0, 1.0)

        ranked_indices = torch.argsort(sims, descending=True).tolist()
        kept: list[tuple[DreamContextEntry, float]] = []
        for idx in ranked_indices:
            s_val = float(sims[idx].item())
            if s_val >= min_similarity or (not kept and len(ranked_indices) > 0):
                kept.append((candidates[idx], s_val))
            if len(kept) >= max(1, top_k):
                break

        top_entry, top_sim = kept[0]
        if top_sim < min_similarity:
            return DreamRetrievalResult(
                has_match=False,
                top_similarity=round(top_sim, 4),
                matched_contexts=(top_entry.user_context,),
                matched_domains=(top_entry.domain,),
                matched_similarities=(round(top_sim, 4),),
                interpolated_z_abs=None,
                recommended_scaffold=top_entry.best_scaffold,
                recommended_policy_text=top_entry.active_policy_text,
            )

        # Concentrate interpolation weight on the top whitened match so unrelated domains do not dilute z_abs
        sim_tensor = torch.tensor(
            [max(0.01, s) * max(0.1, e.best_node_score) for e, s in kept],
            dtype=torch.float32,
            device=target_dev,
        )
        weights = F.softmax(sim_tensor / 0.08, dim=0).unsqueeze(-1)
        z_stack = torch.tensor(
            [e.z_abs for e, _ in kept], dtype=torch.float32, device=target_dev
        )
        z_interp = (weights * z_stack).sum(dim=0, keepdim=True)

        return DreamRetrievalResult(
            has_match=True,
            top_similarity=round(top_sim, 4),
            matched_contexts=tuple(e.user_context for e, _ in kept),
            matched_domains=tuple(e.domain for e, _ in kept),
            matched_similarities=tuple(round(s, 4) for _, s in kept),
            interpolated_z_abs=z_interp,
            recommended_scaffold=top_entry.best_scaffold,
            recommended_policy_text=top_entry.active_policy_text,
        )


