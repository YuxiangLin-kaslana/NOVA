"""Online prototype memories for leakage-free prequential evaluation.

The memory never receives evaluation labels on autonomous reuse events.  A
label is visible only when ``process`` decides to query the external namer (an
oracle in the controlled memory-isolation experiment).  This keeps semantic
discovery cost separate from later, unqueried reuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np


MemoryKey = tuple[Hashable, ...]


@dataclass(frozen=True)
class MemoryConfig:
    name: str
    hierarchical: bool
    radius: float
    merge_radius: float | None = None
    confirm_k: int = 1
    reuse_margin: float = 1.0
    fallback_global: bool = False
    block_label_conflict: bool = False

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.merge_radius is not None and self.merge_radius <= 0:
            raise ValueError("merge_radius must be positive")
        if self.confirm_k < 1:
            raise ValueError("confirm_k must be at least one")
        if not 0 < self.reuse_margin <= 1:
            raise ValueError("reuse_margin must be in (0, 1]")


@dataclass
class Prototype:
    cluster_id: str
    key: MemoryKey
    vector: np.ndarray
    support: int = 1
    verified_support: int = 1
    label_counts: dict[str, int] = field(default_factory=dict)
    committed: bool = True
    created_step: int = 0
    last_step: int = 0

    @property
    def label(self) -> str:
        if not self.label_counts:
            return "unknown"
        return max(self.label_counts, key=lambda label: (self.label_counts[label], label))

    @property
    def verification_consistent(self) -> bool:
        return bool(self.label_counts) and max(self.label_counts.values()) == self.verified_support

    def update(self, vector: np.ndarray, step: int, verified_label: str | None = None) -> None:
        self.support += 1
        eta = 1.0 / self.support
        self.vector = (1.0 - eta) * self.vector + eta * vector
        self.last_step = step
        if verified_label is not None:
            self.verified_support += 1
            self.label_counts[verified_label] = self.label_counts.get(verified_label, 0) + 1


@dataclass(frozen=True)
class MemoryDecision:
    action: str
    pred_label: str
    cluster_id: str | None
    queried: bool
    autonomous_reuse: bool
    created: bool
    distance: float | None
    active_clusters: int
    historical_clusters: int
    merges: int


class OnlinePrototypeMemory:
    """Flat or keyed online prototype memory with optional merge and guard."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._clusters: dict[str, Prototype] = {}
        self._banks: dict[MemoryKey, set[str]] = {}
        self._aliases: dict[str, str] = {}
        self._next_id = 0
        self.historical_clusters = 0
        self.query_count = 0
        self.merge_events: list[dict[str, Any]] = []

    def _bank_key(self, key: MemoryKey) -> MemoryKey:
        return key if self.config.hierarchical else ("__global__",)

    def _resolve(self, cluster_id: str) -> str:
        while cluster_id in self._aliases:
            cluster_id = self._aliases[cluster_id]
        return cluster_id

    def _nearest(
        self,
        vector: np.ndarray,
        bank_key: MemoryKey,
        committed: bool | None = None,
    ) -> tuple[Prototype | None, float]:
        best: Prototype | None = None
        best_distance = float("inf")
        for cluster_id in self._banks.get(bank_key, set()):
            cluster = self._clusters[cluster_id]
            if committed is not None and cluster.committed != committed:
                continue
            distance = float(np.linalg.norm(vector - cluster.vector))
            if distance < best_distance:
                best = cluster
                best_distance = distance
        return best, best_distance

    def _nearest_for_key(
        self,
        vector: np.ndarray,
        bank_key: MemoryKey,
        committed: bool | None = None,
    ) -> tuple[Prototype | None, float]:
        best, best_distance = self._nearest(vector, bank_key, committed=committed)
        if not self.config.fallback_global or best_distance <= self.config.radius:
            return best, best_distance
        global_best: Prototype | None = None
        global_distance = float("inf")
        for cluster in self._clusters.values():
            if committed is not None and cluster.committed != committed:
                continue
            distance = float(np.linalg.norm(vector - cluster.vector))
            if distance < global_distance:
                global_best = cluster
                global_distance = distance
        if global_distance < best_distance:
            return global_best, global_distance
        return best, best_distance

    def _create(self, vector: np.ndarray, bank_key: MemoryKey, label: str, step: int) -> Prototype:
        cluster_id = f"cluster_{self._next_id:05d}"
        self._next_id += 1
        cluster = Prototype(
            cluster_id=cluster_id,
            key=bank_key,
            vector=vector.copy(),
            label_counts={label: 1},
            committed=self.config.confirm_k == 1,
            created_step=step,
            last_step=step,
        )
        self._clusters[cluster_id] = cluster
        self._banks.setdefault(bank_key, set()).add(cluster_id)
        self.historical_clusters += 1
        return cluster

    def _merge_pair(self, first: Prototype, second: Prototype, step: int) -> Prototype:
        if second.support > first.support or (
            second.support == first.support and second.cluster_id < first.cluster_id
        ):
            first, second = second, first

        first_label = first.label
        second_label = second.label
        total = first.support + second.support
        first.vector = (first.support * first.vector + second.support * second.vector) / total
        first.support = total
        first.verified_support += second.verified_support
        first.last_step = step
        for label, count in second.label_counts.items():
            first.label_counts[label] = first.label_counts.get(label, 0) + count
        first.committed = first.committed and second.committed

        self._banks[second.key].remove(second.cluster_id)
        del self._clusters[second.cluster_id]
        self._aliases[second.cluster_id] = first.cluster_id
        self.merge_events.append(
            {
                "step": step,
                "kept": first.cluster_id,
                "removed": second.cluster_id,
                "same_semantic_label": first_label == second_label,
                "first_label": first_label,
                "second_label": second_label,
            }
        )
        return first

    def _merge_until_stable(self, bank_key: MemoryKey, step: int) -> int:
        if self.config.merge_radius is None:
            return 0
        merged = 0
        while True:
            ids = [cluster_id for cluster_id in self._banks.get(bank_key, set()) if self._clusters[cluster_id].committed]
            best_pair: tuple[Prototype, Prototype] | None = None
            best_distance = float("inf")
            for index, first_id in enumerate(ids):
                first = self._clusters[first_id]
                for second_id in ids[index + 1 :]:
                    second = self._clusters[second_id]
                    if self.config.block_label_conflict and first.label != second.label:
                        continue
                    distance = float(np.linalg.norm(first.vector - second.vector))
                    if distance < best_distance:
                        best_pair = (first, second)
                        best_distance = distance
            if best_pair is None or best_distance > self.config.merge_radius:
                break
            self._merge_pair(*best_pair, step=step)
            merged += 1
        return merged

    def process(
        self,
        vector: np.ndarray,
        key: MemoryKey,
        oracle_label: str,
        commit_eligible: bool,
        step: int,
    ) -> MemoryDecision:
        """Process one novelty candidate.

        ``oracle_label`` is consumed only on query branches.  It is accepted as
        an argument so a controlled experiment can reveal the label after the
        decision without a second callback layer.
        """

        bank_key = self._bank_key(key)
        nearest_committed, committed_distance = self._nearest_for_key(vector, bank_key, committed=True)
        nearest_pending, pending_distance = self._nearest_for_key(vector, bank_key, committed=False)
        reuse_threshold = self.config.radius * self.config.reuse_margin

        if nearest_committed is not None and committed_distance <= reuse_threshold:
            pred_label = nearest_committed.label
            nearest_committed.update(vector, step)
            merges = self._merge_until_stable(nearest_committed.key, step)
            return MemoryDecision(
                action="reuse",
                pred_label=pred_label,
                cluster_id=self._resolve(nearest_committed.cluster_id),
                queried=False,
                autonomous_reuse=True,
                created=False,
                distance=committed_distance,
                active_clusters=self.active_count,
                historical_clusters=self.historical_clusters,
                merges=merges,
            )

        self.query_count += 1

        if pending_distance < committed_distance:
            nearest, distance = nearest_pending, pending_distance
        else:
            nearest, distance = nearest_committed, committed_distance

        if not commit_eligible:
            return MemoryDecision(
                action="query_reject",
                pred_label=oracle_label,
                cluster_id=None,
                queried=True,
                autonomous_reuse=False,
                created=False,
                distance=None if nearest is None else distance,
                active_clusters=self.active_count,
                historical_clusters=self.historical_clusters,
                merges=0,
            )

        if nearest is not None and distance <= self.config.radius:
            if nearest.label == oracle_label:
                nearest.update(vector, step, verified_label=oracle_label)
                if not nearest.committed:
                    nearest.committed = (
                        nearest.verified_support >= self.config.confirm_k
                        and nearest.verification_consistent
                    )
                merges = self._merge_until_stable(nearest.key, step)
                action = "query_confirm" if nearest.committed else "query_tentative"
                return MemoryDecision(
                    action=action,
                    pred_label=oracle_label,
                    cluster_id=self._resolve(nearest.cluster_id),
                    queried=True,
                    autonomous_reuse=False,
                    created=False,
                    distance=distance,
                    active_clusters=self.active_count,
                    historical_clusters=self.historical_clusters,
                    merges=merges,
                )

        cluster = self._create(vector, bank_key, oracle_label, step)
        merges = self._merge_until_stable(bank_key, step)
        return MemoryDecision(
            action="query_create" if cluster.committed else "query_create_tentative",
            pred_label=oracle_label,
            cluster_id=self._resolve(cluster.cluster_id),
            queried=True,
            autonomous_reuse=False,
            created=True,
            distance=None if nearest is None else distance,
            active_clusters=self.active_count,
            historical_clusters=self.historical_clusters,
            merges=merges,
        )

    def predict_locked(self, vector: np.ndarray, key: MemoryKey) -> MemoryDecision:
        """Predict from a frozen memory without querying or updating state."""

        bank_key = self._bank_key(key)
        nearest, distance = self._nearest_for_key(vector, bank_key, committed=True)
        threshold = self.config.radius * self.config.reuse_margin
        if nearest is not None and distance <= threshold:
            return MemoryDecision(
                action="reuse_locked",
                pred_label=nearest.label,
                cluster_id=nearest.cluster_id,
                queried=False,
                autonomous_reuse=True,
                created=False,
                distance=distance,
                active_clusters=self.active_count,
                historical_clusters=self.historical_clusters,
                merges=0,
            )
        return MemoryDecision(
            action="unknown_locked",
            pred_label="unknown",
            cluster_id=None,
            queried=False,
            autonomous_reuse=False,
            created=False,
            distance=None if nearest is None else distance,
            active_clusters=self.active_count,
            historical_clusters=self.historical_clusters,
            merges=0,
        )

    @property
    def active_count(self) -> int:
        return len(self._clusters)

    @property
    def committed_count(self) -> int:
        return sum(cluster.committed for cluster in self._clusters.values())

    @property
    def singleton_fraction(self) -> float:
        if not self._clusters:
            return 0.0
        return sum(cluster.support == 1 for cluster in self._clusters.values()) / len(self._clusters)

    @property
    def merge_precision(self) -> float:
        if not self.merge_events:
            return float("nan")
        return float(np.mean([event["same_semantic_label"] for event in self.merge_events]))

    def state(self) -> dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "hierarchical": self.config.hierarchical,
                "radius": self.config.radius,
                "merge_radius": self.config.merge_radius,
                "confirm_k": self.config.confirm_k,
                "reuse_margin": self.config.reuse_margin,
                "fallback_global": self.config.fallback_global,
                "block_label_conflict": self.config.block_label_conflict,
            },
            "active_count": self.active_count,
            "committed_count": self.committed_count,
            "historical_clusters": self.historical_clusters,
            "query_count": self.query_count,
            "singleton_fraction": self.singleton_fraction,
            "merge_precision": self.merge_precision,
            "merge_events": self.merge_events,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "key": list(cluster.key),
                    "support": cluster.support,
                    "verified_support": cluster.verified_support,
                    "label": cluster.label,
                    "label_counts": cluster.label_counts,
                    "committed": cluster.committed,
                    "created_step": cluster.created_step,
                    "last_step": cluster.last_step,
                    "vector": cluster.vector.tolist(),
                }
                for cluster in sorted(self._clusters.values(), key=lambda item: item.cluster_id)
            ],
        }
