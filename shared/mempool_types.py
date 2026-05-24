"""Python mirror of `pallets/quantum-compute-mempool/src/types.rs`.

These dataclasses are the Python side of the `QuantumComputeMempool`
pallet's on-chain types. Field order + tags match the Rust SCALE
representation exactly so substrate-interface can compose calls with them
as `call_params` and so we can decode storage / event payloads back into
typed Python.

Coupled to a specific pallet version: regenerate when the Rust types
change. Parity is exercised by `tests/test_mempool_types.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from shared.ising_feeder import FixedIsingFeeder
from shared.ising_model import IsingModel

if TYPE_CHECKING:
    from shared.miner_types import BlockRequirements


class MinerType(IntEnum):
    """Solver hardware family. Matches `types::MinerType` SCALE enum order."""

    CPU = 0
    GPU = 1
    QPU_DWAVE = 2
    QPU_IBM = 3
    QPU_IONQ = 4
    QPU_PASQAL = 5
    ASIC = 6

    @classmethod
    def from_kind(cls, kind: str) -> "MinerType":
        """Map a `quip-miner` CLI miner kind (cpu/gpu/qpu) to MinerType.

        Used by the `register-solver` subcommand so operators don't need
        to know about the QPU vendor variants — `qpu` defaults to
        QpuDwave to match the legacy CPU/GPU/QPU triplet.
        """
        normalized = kind.lower()
        if normalized == "cpu":
            return cls.CPU
        if normalized == "gpu":
            return cls.GPU
        if normalized in ("qpu", "qpu_dwave", "dwave"):
            return cls.QPU_DWAVE
        if normalized == "qpu_ibm":
            return cls.QPU_IBM
        if normalized == "qpu_ionq":
            return cls.QPU_IONQ
        if normalized == "qpu_pasqal":
            return cls.QPU_PASQAL
        if normalized == "asic":
            return cls.ASIC
        raise ValueError(
            f"unknown miner kind {kind!r}; expected one of "
            "cpu, gpu, qpu, qpu_ibm, qpu_ionq, qpu_pasqal, asic"
        )

    def to_scale_variant(self) -> str:
        """Return the SCALE variant name substrate-interface expects."""
        return {
            MinerType.CPU: "Cpu",
            MinerType.GPU: "Gpu",
            MinerType.QPU_DWAVE: "QpuDwave",
            MinerType.QPU_IBM: "QpuIbm",
            MinerType.QPU_IONQ: "QpuIonq",
            MinerType.QPU_PASQAL: "QpuPasqal",
            MinerType.ASIC: "Asic",
        }[self]

    @classmethod
    def from_scale_variant(cls, name: str) -> "MinerType":
        """Inverse of `to_scale_variant`."""
        for variant in cls:
            if variant.to_scale_variant() == name:
                return variant
        raise ValueError(f"unknown SCALE MinerType variant: {name!r}")


class Formulation(IntEnum):
    """Problem family. v0 supports Ising only."""

    ISING = 0

    def to_scale_variant(self) -> str:
        return "Ising"


class OrderStatus(IntEnum):
    """Lifecycle state of a `JobOrder`."""

    OPENED = 0
    EXPIRED = 1
    CLOSED = 2

    @classmethod
    def from_scale_variant(cls, name: str) -> "OrderStatus":
        try:
            return {"Opened": cls.OPENED, "Expired": cls.EXPIRED, "Closed": cls.CLOSED}[name]
        except KeyError:
            raise ValueError(f"unknown SCALE OrderStatus variant: {name!r}")


# ----------------------------------------------------------------------
# Tagged variant types (RewardResolution, ResultDelivery, JobMode)
# ----------------------------------------------------------------------
# Each is represented as a tagged-union dataclass: a `tag` field holding the
# variant name plus optional fields per variant. The `to_scale_dict` /
# `from_scale_value` helpers convert to/from the dict shapes that
# substrate-interface expects when composing calls or decoding storage.


@dataclass(frozen=True)
class RewardResolution:
    """`SingleBest | TopNWeighted{n} | TopNEqual{n}` enum."""

    tag: str
    n: Optional[int] = None

    @classmethod
    def single_best(cls) -> "RewardResolution":
        return cls(tag="SingleBest")

    @classmethod
    def top_n_weighted(cls, n: int) -> "RewardResolution":
        return cls(tag="TopNWeighted", n=n)

    @classmethod
    def top_n_equal(cls, n: int) -> "RewardResolution":
        return cls(tag="TopNEqual", n=n)

    def to_scale_dict(self) -> dict:
        if self.tag == "SingleBest":
            return {"SingleBest": None}
        if self.tag in ("TopNWeighted", "TopNEqual"):
            if self.n is None:
                raise ValueError(f"{self.tag} requires `n`")
            return {self.tag: {"n": self.n}}
        raise ValueError(f"unknown RewardResolution tag: {self.tag!r}")

    @classmethod
    def from_scale_value(cls, value) -> "RewardResolution":
        if isinstance(value, str):
            if value != "SingleBest":
                raise ValueError(
                    f"unknown bare-string RewardResolution variant: {value!r}; "
                    "only 'SingleBest' has no inner fields"
                )
            return cls(tag=value)
        if isinstance(value, dict) and len(value) == 1:
            (tag, inner), = value.items()
            n = inner.get("n") if isinstance(inner, dict) else None
            return cls(tag=tag, n=n)
        raise ValueError(f"unrecognized RewardResolution shape: {value!r}")


@dataclass(frozen=True)
class ResultDelivery:
    """`OnChainOnly | Callback{endpoint} | CallbackWithPoll{endpoint}` enum."""

    tag: str
    endpoint: Optional[bytes] = None

    @classmethod
    def on_chain_only(cls) -> "ResultDelivery":
        return cls(tag="OnChainOnly")

    @classmethod
    def callback(cls, endpoint: bytes) -> "ResultDelivery":
        return cls(tag="Callback", endpoint=endpoint)

    @classmethod
    def callback_with_poll(cls, endpoint: bytes) -> "ResultDelivery":
        return cls(tag="CallbackWithPoll", endpoint=endpoint)

    def to_scale_dict(self) -> dict:
        if self.tag == "OnChainOnly":
            return {"OnChainOnly": None}
        if self.tag in ("Callback", "CallbackWithPoll"):
            if self.endpoint is None:
                raise ValueError(f"{self.tag} requires `endpoint`")
            return {self.tag: {"endpoint": self.endpoint}}
        raise ValueError(f"unknown ResultDelivery tag: {self.tag!r}")


@dataclass(frozen=True)
class JobMode:
    """`Open | Bid{miners, miner_types}` solver-access policy enum.

    Mode-level filter applied at `submit_solution` time. `Bid` uses OR
    semantics: a solver passes if its account is in `miners` OR its
    solver_type is in `miner_types`.
    """

    tag: str
    miners: Optional[Tuple[bytes, ...]] = None  # AccountId32 list
    miner_types: Optional[Tuple[MinerType, ...]] = None

    @classmethod
    def open(cls) -> "JobMode":
        return cls(tag="Open")

    @classmethod
    def bid(
        cls,
        miners: Optional[Tuple[bytes, ...]] = None,
        miner_types: Optional[Tuple[MinerType, ...]] = None,
    ) -> "JobMode":
        if not miners and not miner_types:
            raise ValueError(
                "Bid mode requires at least one of `miners` or `miner_types` "
                "(pallet rejects EmptyBidCriteria)"
            )
        return cls(tag="Bid", miners=miners, miner_types=miner_types)

    def to_scale_dict(self) -> dict:
        if self.tag == "Open":
            return {"Open": None}
        if self.tag == "Bid":
            miners_field = (
                ["0x" + m.hex() for m in self.miners] if self.miners is not None else None
            )
            miner_types_field = (
                [mt.to_scale_variant() for mt in self.miner_types]
                if self.miner_types is not None
                else None
            )
            return {
                "Bid": {
                    "miners": miners_field,
                    "miner_types": miner_types_field,
                }
            }
        raise ValueError(f"unknown JobMode tag: {self.tag!r}")


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IsingParams:
    """Ising problem definition carried in `JobOrder.ising_params`.

    Matches `types::IsingParams<Nodes, Edges, Fields, Couplings>` field
    order exactly. `h_values` and `j_values` are millivalues (i32) — the
    pallet stores them as fixed-point integers; divide by 1000 to recover
    the float Ising coefficients.

    Quality floors are optional and apply per-submission:
      - `min_energy_milli`: best energy must be ≤ this value (more negative)
      - `min_diversity_milli`: solution diversity must be ≥ this value / 1000
      - `min_solutions`: at least this many valid solutions in the submission
    """

    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    h_values: Tuple[int, ...]      # i32 millivalues
    j_values: Tuple[int, ...]      # i32 millivalues, len == len(edges)
    min_energy_milli: Optional[int] = None
    min_diversity_milli: Optional[int] = None
    min_solutions: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.h_values) != len(self.nodes):
            raise ValueError(
                f"h_values length ({len(self.h_values)}) must match nodes "
                f"length ({len(self.nodes)})"
            )
        if len(self.j_values) != len(self.edges):
            raise ValueError(
                f"j_values length ({len(self.j_values)}) must match edges "
                f"length ({len(self.edges)})"
            )

    def to_scale_dict(self) -> dict:
        """Render as the dict shape substrate-interface expects.

        BoundedVec fields encode as 1-element tuples (`(value,)`) when the
        composite is metadata-exposed as a single-field struct — same rule
        Phase 4 hit with `register_topology`.
        """
        return {
            "nodes": (list(self.nodes),),
            "edges": (list(self.edges),),
            "h_values": (list(self.h_values),),
            "j_values": (list(self.j_values),),
            "min_energy_milli": self.min_energy_milli,
            "min_diversity_milli": self.min_diversity_milli,
            "min_solutions": self.min_solutions,
        }

    @classmethod
    def from_scale_value(cls, value: dict) -> "IsingParams":
        """Decode from a substrate-interface storage query value."""
        return cls(
            nodes=tuple(int(n) for n in value["nodes"]),
            edges=tuple((int(u), int(v)) for u, v in value["edges"]),
            h_values=tuple(int(h) for h in value["h_values"]),
            j_values=tuple(int(j) for j in value["j_values"]),
            min_energy_milli=(
                int(value["min_energy_milli"])
                if value.get("min_energy_milli") is not None
                else None
            ),
            min_diversity_milli=(
                int(value["min_diversity_milli"])
                if value.get("min_diversity_milli") is not None
                else None
            ),
            min_solutions=(
                int(value["min_solutions"])
                if value.get("min_solutions") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class OrderTiming:
    deadline_blocks: int
    block_wait: int

    def to_scale_dict(self) -> dict:
        return {
            "deadline_blocks": self.deadline_blocks,
            "block_wait": self.block_wait,
        }


@dataclass(frozen=True)
class MempoolSolverInfo:
    """Storage-side `SolverInfo` for a registered solver.

    Named with the `Mempool` prefix to avoid colliding with `MinerInfo`
    from `substrate.types` (which describes a PoW miner).
    """

    account: bytes
    solver_type: MinerType
    registered_at: int
    solutions_submitted: int
    rewards_earned: int  # Balance, u128


@dataclass(frozen=True)
class JobOrder:
    """Decoded `JobOrders[order_id]` storage entry."""

    spec_id: bytes
    proposer: bytes
    ising_params: IsingParams
    reward: int
    mode: JobMode
    resolution: RewardResolution
    timing: OrderTiming
    delivery: ResultDelivery
    status: OrderStatus
    created_at: int
    first_solution_at: Optional[int]
    solution_count: int


@dataclass(frozen=True)
class MempoolJobContext:
    """Work context for a mempool-side mining task.

    Mirrors the shape of `SubstrateMiningContext` (PoW side) in the fields
    the mining loop needs (`nodes`, `edges`, plus a way to materialize the
    Ising problem), but carries mempool-specific identity (`order_id`)
    and the chain-encoded quality floors from `IsingParams`.

    Construction is normally `MempoolJobContext.from_job_order(order_id,
    order)` — see Phase 8c's controller for how the order is fetched.
    """

    order_id: int
    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    h_values: Tuple[int, ...]   # i32 millivalues per node
    j_values: Tuple[int, ...]   # i32 millivalues per edge
    min_energy_milli: Optional[int] = None
    min_diversity_milli: Optional[int] = None
    min_solutions: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.h_values) != len(self.nodes):
            raise ValueError(
                f"h_values length ({len(self.h_values)}) must match nodes "
                f"length ({len(self.nodes)})"
            )
        if len(self.j_values) != len(self.edges):
            raise ValueError(
                f"j_values length ({len(self.j_values)}) must match edges "
                f"length ({len(self.edges)})"
            )

    @classmethod
    def from_job_order(cls, order_id: int, order: "JobOrder") -> "MempoolJobContext":
        ising = order.ising_params
        return cls(
            order_id=order_id,
            nodes=ising.nodes,
            edges=ising.edges,
            h_values=ising.h_values,
            j_values=ising.j_values,
            min_energy_milli=ising.min_energy_milli,
            min_diversity_milli=ising.min_diversity_milli,
            min_solutions=ising.min_solutions,
        )

    def resolve_ising(
        self,
        salt: bytes,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
    ) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int]:
        """Map chain-carried (h_values, j_values) directly. Salt and nodes unused."""
        h = {
            int(node): float(hv) / 1000.0
            for node, hv in zip(self.nodes, self.h_values)
        }
        J = {
            (int(edge[0]), int(edge[1])): float(jv) / 1000.0
            for edge, jv in zip(self.edges, self.j_values)
        }
        return h, J, 0  # 0 = placeholder nonce for telemetry

    def make_feeder(
        self,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
        *,
        buffer_size: int = 8,
    ) -> FixedIsingFeeder:
        """Return a ``FixedIsingFeeder`` wrapping the order's stored (h, J).

        Mempool work uses pre-baked Ising parameters straight off the
        chain, so no derivation is needed. The placeholder ``nonce`` /
        ``salt`` bytes mirror what :meth:`resolve_ising` returned in the
        old inline path — telemetry only; the chain doesn't re-derive
        them. ``nodes`` / ``edges`` mirror ``WorkContext.make_feeder``'s
        signature but go unused here because the model is already
        finalised against the chain-stored node/edge order.

        ``buffer_size`` is accepted for API parity with
        :class:`RandomIsingFeeder` and intentionally ignored — the
        fixed model list has fixed length.
        """
        del nodes, edges, buffer_size  # See docstring.
        h = {
            int(node): float(hv) / 1000.0
            for node, hv in zip(self.nodes, self.h_values)
        }
        J = {
            (int(edge[0]), int(edge[1])): float(jv) / 1000.0
            for edge, jv in zip(self.edges, self.j_values)
        }
        model = IsingModel(
            h=h,
            J=J,
            nonce=b"\x00" * 32,  # placeholder; chain doesn't re-derive
            salt=b"\x00" * 32,   # placeholder; salt unused on this path
        )
        return FixedIsingFeeder(models=[model])

    def requirements(self) -> "BlockRequirements":
        """Build BlockRequirements from this job's quality floors."""
        from shared.miner_types import BlockRequirements

        difficulty_energy = (
            float(self.min_energy_milli) / 1000.0
            if self.min_energy_milli is not None
            else float("inf")
        )
        min_diversity = (
            float(self.min_diversity_milli) / 1000.0
            if self.min_diversity_milli is not None
            else 0.0
        )
        min_solutions = (
            int(self.min_solutions) if self.min_solutions is not None else 1
        )
        return BlockRequirements(
            difficulty_energy=difficulty_energy,
            min_diversity=min_diversity,
            min_solutions=min_solutions,
            timeout_to_difficulty_adjustment_decay=2**31 - 1,
        )


# ----------------------------------------------------------------------
# Event dataclasses (a subset — enough for the Phase 8 controller's
# subscription path; add more as they're consumed).
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class JobProposedEvent:
    order_id: int
    spec_id: bytes
    proposer: bytes
    reward: int
    deadline_blocks: int
    block_wait: int


@dataclass(frozen=True)
class SolutionAcceptedEvent:
    order_id: int
    solver: bytes
    energy_milli: int
    diversity_milli: int


@dataclass(frozen=True)
class OrderExpiredEvent:
    order_id: int


@dataclass(frozen=True)
class RewardClaimedEvent:
    order_id: int
    solver: bytes
    amount: int


# ----------------------------------------------------------------------
# Helpers for callers building `propose_job` params
# ----------------------------------------------------------------------


def solutions_to_scale(solutions: List[List[int]]) -> List[List[int]]:
    """Coerce a Python list-of-lists of spins to the i8 SCALE shape.

    Each inner list must contain only -1, 0, or +1; the pallet rejects
    other values via `InvalidSpinValues`. Returns the same shape — present
    here so callers can flag malformed input *before* RPC submission.
    """
    out: List[List[int]] = []
    for i, sol in enumerate(solutions):
        for j, s in enumerate(sol):
            if s not in (-1, 0, 1):
                raise ValueError(
                    f"solutions[{i}][{j}] = {s}; only -1, 0, +1 allowed"
                )
        out.append(list(sol))
    return out


__all__ = [
    "Formulation",
    "IsingParams",
    "JobMode",
    "JobOrder",
    "JobProposedEvent",
    "MempoolJobContext",
    "MempoolSolverInfo",
    "MinerType",
    "OrderExpiredEvent",
    "OrderStatus",
    "OrderTiming",
    "ResultDelivery",
    "RewardClaimedEvent",
    "RewardResolution",
    "SolutionAcceptedEvent",
    "solutions_to_scale",
]
