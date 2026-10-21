"""Run Fast BO on one combinatorial benchmark and seed, using our patched MCBO.

Example:
    python run_comb.py --task labs --acquisition ucb --seed 42

The paper uses seeds 42-61. Results are saved like in run.py, with points as category
indices and values minimized (MCBO's convention).
"""

import os

# We used a single thread for the paper
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

MCBO_DIR = Path(__file__).resolve().parent / "third_party" / "HEBO" / "MCBO"
sys.path.insert(0, str(MCBO_DIR))
from mcbo.utils.experiment_utils import get_opt, get_task_from_id  # noqa: E402

N_INIT, N_ITER = 20, 180
# MCBO optimizer IDs: model__acq_optimizer__acq_func__trust_region
METHODS = {
    "ucb": "lr_ol__fast_mm__lcb__basic",
    "ei": "lr_ol__fast_mm__ei__basic",
    "ts": "lr_ol__fast_ts__ts__basic",
}
TASKS = ["contamination", "labs", "maxsat60", "maxsat125", "pest"]


def main() -> None:
    """Run the BO loop and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--acquisition", required=True, choices=list(METHODS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = get_task_from_id(task_id=args.task)
    opt = get_opt(task=task, short_opt_id=METHODS[args.acquisition], bo_n_init=N_INIT,
                  dtype=torch.float64, bo_device=torch.device("cpu"))

    # Random initial design, the trust region starts at the best point
    x0 = opt.search_space.sample(N_INIT)
    y0 = task(x0)
    opt.initialize(x0, y0)
    if opt.tr_manager.center is None:
        opt.tr_manager.set_center(opt.data_buffer.best_x)

    xs = [opt.search_space.transform(x0).cpu().numpy()]
    ys = [np.asarray(y0).reshape(-1)]
    times = {name: np.empty(N_ITER) for name in ("ask", "tell", "eval")}
    for i in range(N_ITER):
        start = time.perf_counter()
        x = opt.method_suggest_transformed(1)
        # MCBO's suggest() draws a throwaway sample to read column dtypes. We draw it
        # too, so the run matches MCBO's DataFrame interface exactly.
        opt.search_space.consume_sample_rng(1)
        times["ask"][i] = time.perf_counter() - start

        start = time.perf_counter()
        y = task.evaluate_transformed(x.cpu().numpy())
        times["eval"][i] = time.perf_counter() - start

        start = time.perf_counter()
        opt.method_observe_transformed(x, y)
        times["tell"][i] = time.perf_counter() - start
        xs.append(x.cpu().numpy())
        ys.append(np.asarray(y).reshape(-1))

    method = f"fast_{args.acquisition}"
    out = args.out / args.task / method
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"seed_{args.seed}_xs.npy", np.concatenate(xs).astype(np.int64))
    ys = np.concatenate(ys).astype(np.float64)
    np.save(out / f"seed_{args.seed}_ys.npy", ys)
    np.savez(out / f"seed_{args.seed}_times.npz", **times)
    algorithm_time = times["ask"].sum() + times["tell"].sum()
    print(f"Best value {ys.min():.4f}; algorithm time {algorithm_time:.2f} s")


if __name__ == "__main__":
    main()
