"""Run Fast BO on one continuous benchmark and seed.

Example:
    python run.py --benchmark svm --acquisition ucb --seed-idx 0

Saves all evaluated points and values to
``results/<benchmark>/<method>/seed_<seed>_{xs,ys}.npy``, and the ask, eval and tell
times of each iteration to ``seed_<seed>_times.npz``.
"""

import os

# We used 4 threads for the paper (BLAS results depend on the thread count)
N_THREADS = 4
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(N_THREADS))

import argparse  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from fast_bo.agent import FastLinearBO  # noqa: E402
from fast_bo.benchmarks import BENCHMARKS, BencherBenchmark  # noqa: E402
from fast_bo.low_dim import FastLinearBOLowDim  # noqa: E402

N_INITIAL_POINTS = 30
# Seed i is child i of SeedSequence(0). The paper uses 10 seeds high-D, 40 low-D.
SEEDS = [int(c.generate_state(1)[0]) for c in np.random.SeedSequence(0).spawn(40)]


def generate_seed(seed_seq: np.random.SeedSequence) -> int:
    """Draw an integer seed from a SeedSequence."""
    return int(seed_seq.generate_state(1)[0])


def seed_everything(seed: int) -> None:
    """Seed the global random number generators."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(
    benchmark: BencherBenchmark, agent: FastLinearBO, seed: int, batch_size: int = 1
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Run the ask-tell loop from a Sobol initial design until the budget is used.

    All randomness comes from ``seed`` through a SeedSequence tree, with separate
    branches for the initial design, the loop and the evaluation seeds.

    Args:
        benchmark: Benchmark to optimize.
        agent: Optimizer.
        seed: Run seed.
        batch_size: Number of points per iteration.

    Returns:
        Evaluated points, values and per-iteration times.
    """
    info = benchmark.info
    init_key, loop_key, eval_key = np.random.SeedSequence(seed).spawn(3)
    gen_init_key, tell_init_key = init_key.spawn(2)

    # Noisy benchmarks use either one fixed set of rollout seeds or a new set per point
    fixed_seed = (
        generate_seed(eval_key.spawn(1)[0]) if info.fixed_evaluation_seeds else None
    )

    def evaluation_seeds(n_points: int) -> list[int]:
        if fixed_seed is not None:
            return [fixed_seed] * n_points
        return [generate_seed(child) for child in eval_key.spawn(n_points)]

    # Initial design
    seed_init = generate_seed(gen_init_key)
    seed_everything(seed_init)
    sobol = qmc.Sobol(
        d=info.input_dim, scramble=True, rng=np.random.default_rng(seed_init)
    )
    lower, upper = np.zeros(info.input_dim), np.ones(info.input_dim)
    x = qmc.scale(sobol.random(n=N_INITIAL_POINTS), lower, upper).astype(np.float32)
    x = np.clip(x, lower.astype(np.float32), upper.astype(np.float32))
    y = benchmark.evaluate_points(x, evaluation_seeds(len(x))).astype(np.float32)
    seed_tell = generate_seed(tell_init_key)
    seed_everything(seed_tell)
    agent.tell_initial_data(x, y, seed_tell)

    times: dict[str, list[float]] = {"ask": [], "tell": [], "eval": []}
    while len(y) < info.n_evaluations:
        ask_key, tell_key = loop_key.spawn(1)[0].spawn(2)

        seed_ask = generate_seed(ask_key)
        seed_everything(seed_ask)
        start = time.perf_counter()
        new_x = agent.ask(min(batch_size, info.n_evaluations - len(y)), seed_ask)
        times["ask"].append(time.perf_counter() - start)
        new_x = np.clip(
            new_x.astype(np.float32), lower.astype(np.float32), upper.astype(np.float32)
        )

        start = time.perf_counter()
        new_y = benchmark.evaluate_points(new_x, evaluation_seeds(len(new_x)))
        new_y = new_y.astype(np.float32)
        times["eval"].append(time.perf_counter() - start)
        x, y = np.concatenate((x, new_x)), np.concatenate((y, new_y))

        seed_tell = generate_seed(tell_key)
        seed_everything(seed_tell)
        start = time.perf_counter()
        agent.tell(new_x, new_y, seed_tell)
        times["tell"].append(time.perf_counter() - start)

        print(f"{len(y):5d} evaluations, best {y.max():.4f}", flush=True)
    return x, y, times


def main() -> None:
    """Parse the arguments, run, and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--acquisition", required=True, choices=["ucb", "ei", "ts"])
    parser.add_argument("--seed-idx", type=int, required=True, choices=range(40))
    parser.add_argument("--address", default="127.0.0.1", help="Bencher server")
    parser.add_argument("--port", type=int, default=50051, help="Bencher port")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(N_THREADS)

    benchmark = BencherBenchmark(args.benchmark, address=args.address, port=args.port)
    low_dim = benchmark.info.low_dim
    agent_class = FastLinearBOLowDim if low_dim else FastLinearBO
    agent = agent_class(
        input_dim=benchmark.info.input_dim, acquisition=args.acquisition
    )
    method = f"fast_linear_{args.acquisition}" + ("_low_dim" if low_dim else "")
    seed = SEEDS[args.seed_idx]

    out = args.out / args.benchmark / method
    out.mkdir(parents=True, exist_ok=True)
    x, y, times = run(benchmark, agent, seed)
    np.save(out / f"seed_{seed}_xs.npy", x)
    np.save(out / f"seed_{seed}_ys.npy", y)
    np.savez(out / f"seed_{seed}_times.npz", **times)
    algorithm_time = sum(times["ask"]) + sum(times["tell"])
    print(f"Best value {y.max():.4f}; algorithm time {algorithm_time:.1f} s")


if __name__ == "__main__":
    main()
