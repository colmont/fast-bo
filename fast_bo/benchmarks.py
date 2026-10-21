"""Continuous benchmarks, served by Bencher."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value, ValueType
from jaxtyping import Float


@dataclass(frozen=True)
class BenchmarkInfo:
    """Benchmark settings.

    Args:
        input_dim: Input dimension (the domain is [0,1]^d).
        n_evaluations: Total budget, including the initial design.
        low_dim: Use the trust-region variant.
        n_evaluation_seeds: Number of seeded rollouts averaged per point.
        fixed_evaluation_seeds: Use the same rollout seeds for every point.
    """

    input_dim: int
    n_evaluations: int
    low_dim: bool = False
    n_evaluation_seeds: int = 1
    fixed_evaluation_seeds: bool = False


BENCHMARKS: dict[str, BenchmarkInfo] = {
    # Low-dimensional
    "lunarlander": BenchmarkInfo(
        12, 500, low_dim=True, n_evaluation_seeds=50, fixed_evaluation_seeds=True
    ),
    "robotpushing": BenchmarkInfo(14, 500, low_dim=True),
    "mujoco-swimmer": BenchmarkInfo(16, 500, low_dim=True, n_evaluation_seeds=10),
    "mujoco-hopper": BenchmarkInfo(33, 500, low_dim=True, n_evaluation_seeds=10),
    # High-dimensional
    "rover": BenchmarkInfo(60, 1000),
    "mopta08": BenchmarkInfo(124, 1000),
    "lasso-dna": BenchmarkInfo(180, 1000),
    "svm": BenchmarkInfo(388, 1000),
    "mujoco-ant": BenchmarkInfo(888, 1000, n_evaluation_seeds=10),
    "mujoco-humanoid": BenchmarkInfo(6392, 1000, n_evaluation_seeds=10),
}


class BencherBenchmark:
    """Client for a benchmark on a Bencher server. Values are negated (maximization).

    With our patch, Bencher runs one seeded rollout per call for MuJoCo and Lunar
    Lander, and we average `n_evaluation_seeds` of them here.

    Args:
        name: Benchmark name, a key of `BENCHMARKS`.
        address: Server address, a hostname or a "unix://" socket path.
        port: Server port (ignored for Unix sockets).
    """

    def __init__(
        self, name: str, address: str = "127.0.0.1", port: int = 50051
    ) -> None:
        self.name = name
        self.info = BENCHMARKS[name]
        self.client = BencherClient(address=address, port=port)

    def evaluate_points(
        self, points: Float[np.ndarray, "N D"], seeds: Sequence[int]
    ) -> Float[np.ndarray, " N"]:
        """Evaluate points, averaging `n_evaluation_seeds` rollouts per point.

        Args:
            points: Points to evaluate.
            seeds: One seed per point.

        Returns:
            Negated values (higher is better).
        """
        n_rollouts = self.info.n_evaluation_seeds
        evaluations = []
        for point, seed in zip(points, seeds, strict=True):
            rollout_seeds = [
                int(child.generate_state(1)[0])
                for child in np.random.SeedSequence(seed).spawn(n_rollouts)
            ]
            bencher_point = [
                Value(type=ValueType.CONTINUOUS, value=float(d)) for d in point
            ]
            values = [
                self.client.evaluate_point(
                    benchmark_name=self.name,
                    point=bencher_point,
                    random_seed=rollout_seed,
                )
                .objectives[0]
                .value
                for rollout_seed in rollout_seeds
            ]
            evaluations.append(float(np.mean(values)))
        return -np.array(evaluations)  # Bencher minimizes; we maximize
