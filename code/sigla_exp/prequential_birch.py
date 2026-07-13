"""Causal Online BIRCH baseline with stable semantic cluster identifiers."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from sigla_exp.prequential_memory import MemoryDecision

try:
    from sklearn.cluster import Birch
except ImportError as exc:  # pragma: no cover - exercised by the runner environment check
    Birch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class StableSubcluster:
    stable_id: str
    center: np.ndarray
    label: str | None = None
    support: int = 0


class OnlineBirchMemory:
    """Incremental BIRCH with causal center matching across tree updates.

    Scikit-learn's subcluster indices are not stable after ``partial_fit``.
    This wrapper matches new centers to old centers using geometry only; true
    labels are attached solely when the discovery policy explicitly queries.
    """

    def __init__(self, threshold: float, branching_factor: int = 50):
        if Birch is None:
            raise RuntimeError("scikit-learn is required for OnlineBirchMemory") from _IMPORT_ERROR
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.match_radius = 2.0 * self.threshold
        self.model = Birch(
            threshold=self.threshold,
            branching_factor=branching_factor,
            n_clusters=None,
            compute_labels=False,
        )
        self._fitted = False
        self._clusters: dict[str, StableSubcluster] = {}
        self._next_id = 0
        self.historical_clusters = 0
        self.query_count = 0

    def _new_id(self) -> str:
        stable_id = f"birch_{self._next_id:05d}"
        self._next_id += 1
        self.historical_clusters += 1
        return stable_id

    def _nearest(self, vector: np.ndarray) -> tuple[StableSubcluster | None, float]:
        best: StableSubcluster | None = None
        best_distance = float("inf")
        for cluster in self._clusters.values():
            distance = float(np.linalg.norm(vector - cluster.center))
            if distance < best_distance:
                best = cluster
                best_distance = distance
        return best, best_distance

    def _refresh_centers(self) -> None:
        centers = np.asarray(self.model.subcluster_centers_, dtype=np.float64)
        old = list(self._clusters.values())
        pairs: list[tuple[float, int, int]] = []
        for old_index, cluster in enumerate(old):
            for new_index, center in enumerate(centers):
                pairs.append((float(np.linalg.norm(cluster.center - center)), old_index, new_index))
        pairs.sort()

        used_old: set[int] = set()
        used_new: set[int] = set()
        refreshed: dict[str, StableSubcluster] = {}
        for distance, old_index, new_index in pairs:
            if distance > self.match_radius:
                break
            if old_index in used_old or new_index in used_new:
                continue
            previous = old[old_index]
            refreshed[previous.stable_id] = StableSubcluster(
                stable_id=previous.stable_id,
                center=centers[new_index].copy(),
                label=previous.label,
                support=previous.support,
            )
            used_old.add(old_index)
            used_new.add(new_index)

        for new_index, center in enumerate(centers):
            if new_index in used_new:
                continue
            stable_id = self._new_id()
            refreshed[stable_id] = StableSubcluster(stable_id=stable_id, center=center.copy())
        self._clusters = refreshed

    def _partial_fit(self, vector: np.ndarray) -> None:
        self.model.partial_fit(vector[None, :])
        self._fitted = True
        self._refresh_centers()

    def process(
        self,
        vector: np.ndarray,
        oracle_label: str,
        commit_eligible: bool,
        step: int,
    ) -> MemoryDecision:
        del step
        trial = copy.deepcopy(self)
        trial._partial_fit(vector)
        assigned, assigned_distance = trial._nearest(vector)
        if assigned is None:
            raise RuntimeError("BIRCH produced no subcluster after partial_fit")

        if assigned.label is not None:
            pred_label = assigned.label
            stable_id = assigned.stable_id
            assigned.support += 1
            self.__dict__.update(trial.__dict__)
            return MemoryDecision(
                action="reuse",
                pred_label=pred_label,
                cluster_id=stable_id,
                queried=False,
                autonomous_reuse=True,
                created=False,
                distance=assigned_distance,
                active_clusters=self.active_count,
                historical_clusters=self.historical_clusters,
                merges=0,
            )

        self.query_count += 1
        if not commit_eligible:
            return MemoryDecision(
                action="query_reject",
                pred_label=oracle_label,
                cluster_id=None,
                queried=True,
                autonomous_reuse=False,
                created=False,
                distance=assigned_distance,
                active_clusters=self.active_count,
                historical_clusters=self.historical_clusters,
                merges=0,
            )

        assigned.label = oracle_label
        assigned.support += 1
        trial.query_count = self.query_count
        self.__dict__.update(trial.__dict__)
        return MemoryDecision(
            action="query_create",
            pred_label=oracle_label,
            cluster_id=assigned.stable_id,
            queried=True,
            autonomous_reuse=False,
            created=True,
            distance=assigned_distance,
            active_clusters=self.active_count,
            historical_clusters=self.historical_clusters,
            merges=0,
        )

    def predict_locked(self, vector: np.ndarray) -> MemoryDecision:
        nearest, distance = self._nearest(vector)
        if nearest is not None and nearest.label is not None:
            return MemoryDecision(
                action="reuse_locked",
                pred_label=nearest.label,
                cluster_id=nearest.stable_id,
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
        return sum(cluster.label is not None for cluster in self._clusters.values())

    @property
    def singleton_fraction(self) -> float:
        if not self._clusters:
            return 0.0
        return sum(cluster.support <= 1 for cluster in self._clusters.values()) / len(self._clusters)

    @property
    def merge_precision(self) -> float:
        return float("nan")

    def state(self) -> dict[str, object]:
        return {
            "active_count": self.active_count,
            "committed_count": self.committed_count,
            "historical_clusters": self.historical_clusters,
            "query_count": self.query_count,
            "singleton_fraction": self.singleton_fraction,
            "clusters": [
                {
                    "cluster_id": cluster.stable_id,
                    "label": cluster.label,
                    "support": cluster.support,
                    "center": cluster.center.tolist(),
                }
                for cluster in sorted(self._clusters.values(), key=lambda item: item.stable_id)
            ],
        }
