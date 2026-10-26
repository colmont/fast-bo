# Fast BO

Code for the paper *Fast Bayesian Optimization Through Linear Models* by Colin Doumont,
Henry Moss and Philipp Hennig ([arXiv:2610.29580](https://arxiv.org/abs/2610.29580)).

The code for continuous problems is in `fast_bo/`. For categorical problems we patch
MCBO (`patches/mcbo.patch`).

## Continuous benchmarks

The benchmarks come from [Bencher](https://github.com/lpapenme/bencher), which runs in
Docker. Set it up with [uv](https://docs.astral.sh/uv/):

```bash
./setup_bencher.sh
docker run -p 50051:50051 -d bencher-fast-bo
uv sync
```

To run one benchmark with one acquisition function (`ucb`, `ei` or `ts`) and one seed:

```bash
uv run python run.py --benchmark svm --acquisition ucb --seed-idx 0
```

The benchmarks are `lunarlander`, `robotpushing`, `mujoco-swimmer` and `mujoco-hopper`
(low-dimensional, seeds 0-39), and `rover`, `mopta08`, `lasso-dna`, `svm`, `mujoco-ant`
and `mujoco-humanoid` (high-dimensional, seeds 0-9). Results are saved to `results/`.
For each run you get the evaluated points, their values and the time per iteration.

## Combinatorial benchmarks

```bash
cd combinatorial
./setup_mcbo.sh
uv run python run_comb.py --task labs --acquisition ucb --seed 42
```

`setup_mcbo.sh` fetches [MCBO](https://github.com/huawei-noah/HEBO/tree/master/MCBO),
applies our patch and sets up the environment. The tasks are `contamination`, `labs`,
`maxsat60`, `maxsat125` and `pest`, and the paper uses seeds 42-61. MCBO minimizes, so
the saved values are minimized too.

## Credits

The LABS, MaxSAT and Contamination tasks are adapted from
[Bounce](https://github.com/lpapenme/bounce). The trust region follows
[TuRBO](https://github.com/uber-research/TuRBO). MIT License.

## Citation

```bibtex
@article{doumont2026fast,
  title   = {Fast {B}ayesian Optimization Through Linear Models},
  author  = {Doumont, Colin and Moss, Henry and Hennig, Philipp},
  journal = {arXiv preprint arXiv:2610.29580},
  year    = {2026},
}
```
