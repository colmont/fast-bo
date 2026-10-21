"""Fast BO inside a trust region, for low-dimensional problems."""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from jaxtyping import Float
from linear_operator.operators import (
    LowRankRootLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
)
from torch import nn
from torch.quasirandom import SobolEngine

from fast_bo.agent import FastLinearBO, project_onto_unit_sphere


class SphericalLinearKernel(Kernel):
    """Linear BO's spherical linear kernel, with ARD.

    Only used to fit the ARD lengthscales that shape the trust region:

        k(x, x') = b0 + b1 * <P(z), P(z')>

    where z is the centered input divided by the ARD and global lengthscales.

    Args:
        ard_num_dims: Input dimension.
        kwargs: Passed on to gpytorch's Kernel.
    """

    has_lengthscale = True

    def __init__(self, ard_num_dims: int, **kwargs: object) -> None:
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

        # (b0, b1) = softmax(raw_coeffs)
        self.raw_coeffs = nn.Parameter(torch.zeros(*self.batch_shape, 2))

        # Global lengthscale: sqrt(sigmoid(raw_glob_ls) * max_sq_norm)
        self.raw_glob_ls = nn.Parameter(torch.zeros(*self.batch_shape, 1))

        # ARD lengthscales: log-normal prior, lower bound 0.025
        ls_prior = LogNormalPrior(
            loc=math.sqrt(2),
            scale=math.sqrt(3),
        )
        ls_constraint = GreaterThan(0.025, transform=None, initial_value=ls_prior.mode)
        self.register_prior(
            "lengthscale_prior",
            ls_prior,
            lambda m: m.lengthscale,
            lambda m, v: m._set_lengthscale(v),  # noqa: SLF001
        )
        self.register_constraint("raw_lengthscale", ls_constraint)

        self._mins: torch.Tensor | None = None
        self._maxs: torch.Tensor | None = None
        self._centers: torch.Tensor | None = None

    def set_bounds(self, bounds: Float[torch.Tensor, "2 D"]) -> None:
        """Set the input bounds (used for centering and the global lengthscale).

        Args:
            bounds: Lower and upper bounds, shape (2, d).
        """
        self._mins = bounds[0]
        self._maxs = bounds[1]
        self._centers = (bounds[0] + bounds[1]) / 2.0

    def forward(
        self,
        x1: Float[torch.Tensor, "... N D"],
        x2: Float[torch.Tensor, "... M D"],
        diag: bool = False,  # noqa: FBT002
        **params: object,  # noqa: ARG002
    ) -> (
        RootLinearOperator
        | LowRankRootLinearOperator
        | MatmulLinearOperator
        | Float[torch.Tensor, "... N"]
    ):
        """Compute the kernel matrix (the diagonal is all ones)."""
        if diag:
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)

        x1_equal_x2 = torch.equal(x1, x2)
        lengthscale = self.lengthscale  # (1, D) with ARD

        # Global lengthscale, relative to the largest possible squared norm
        max_sq_norm = (
            ((self._maxs - self._mins) / (2 * lengthscale))
            .square()
            .sum(dim=-1, keepdim=True)
        )
        glob_ls = torch.sqrt(torch.sigmoid(self.raw_glob_ls[..., None]) * max_sq_norm)

        coeffs = torch.softmax(self.raw_coeffs, dim=-1)
        sqrt_const = torch.sqrt(coeffs[..., 0])
        sqrt_linear = torch.sqrt(coeffs[..., 1])

        def _featurize(x: torch.Tensor) -> torch.Tensor:
            x_ = (x - self._centers).div(lengthscale).div(glob_ls)
            x_ = project_onto_unit_sphere(x_)
            return torch.cat(
                [
                    x_ * sqrt_linear[..., None, None],
                    sqrt_const[..., None, None].expand(*x_.shape[:-1], 1),
                ],
                dim=-1,
            )

        x1_ = _featurize(x1)
        if x1_equal_x2:
            n, d = x1_.shape[-2:]
            if d >= n:
                return RootLinearOperator(x1_)
            return LowRankRootLinearOperator(x1_)
        return MatmulLinearOperator(x1_, _featurize(x2).mT)


@dataclass
class TuRBOState:
    """Trust region state, as in TuRBO."""

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = dataclasses.field(init=False)  # set in __post_init__
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        """Set the failure tolerance."""
        # See: https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_1.py#L106
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, self.dim / self.batch_size)
        )


def update_state(state: TuRBOState, y_next: Float[torch.Tensor, "n 1"]) -> TuRBOState:
    """Grow or shrink the trust region after a batch (in place).

    Args:
        state: Current TuRBO state.
        y_next: Values of the latest batch.

    Returns:
        The updated state.
    """
    # See: https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_1.py#L138
    if max(y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


class FastLinearBOLowDim(FastLinearBO):
    """Fast BO inside a TuRBO trust region.

    Same trust region as TuRBO: centered on the best point, it grows after successes,
    shrinks after failures, and restarts from a new Sobol design once it gets too
    small. Its side lengths are scaled by ARD lengthscales from Linear BO's kernel,
    refit every `ard_refit_period` fits. Fast BO itself is only fit on the last D
    points.

    Reference:
        Eriksson, D., et al. (2019). Scalable Global Optimization via Local Bayesian
        Optimization. Neural Information Processing Systems (NeurIPS).
    """

    ard_refit_period = 50

    def tell_initial_data(
        self, x: Float[np.ndarray, "N D"], y: Float[np.ndarray, " N"], seed: int
    ) -> None:
        """Set the initial data, start the trust region and fit.

        Args:
            x: Initial points.
            y: Observed values.
            seed: Unused.
        """
        self.n_init = len(y)  # size of the Sobol design after a restart
        self.state = TuRBOState(
            dim=self.input_dim, batch_size=1, best_value=float(np.max(y))
        )
        self._restart_buffer = torch.empty(
            0, self.input_dim, dtype=self.dtype, device=self.device
        )
        self._ard_weights = None
        self._n_fits = 0
        super().tell_initial_data(x, y, seed)

    def tell(
        self, x: Float[np.ndarray, "N D"], y: Float[np.ndarray, " N"], seed: int
    ) -> None:
        """Update the trust region, then refit or restart.

        Args:
            x: New points.
            y: Observed values.
            seed: Seed for the Sobol design after a restart.
        """
        new_X = torch.tensor(x, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        # Points from a restart design do not count as successes or failures
        if self.fraction is not None:
            self.state = update_state(self.state, new_y)
        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])

        if self.state.restart_triggered:
            self.state = TuRBOState(
                dim=self.input_dim, batch_size=self.state.batch_size
            )
            self.train_X = self.train_X[:0]
            self.train_y = self.train_y[:0]
            self.fraction, self._fits_since_refresh = None, 0
            self._ard_weights, self._n_fits = None, 0
            sobol = SobolEngine(self.input_dim, scramble=True, seed=seed)
            self._restart_buffer = sobol.draw(self.n_init).to(
                dtype=self.dtype, device=self.device
            )
        elif self._restart_buffer.shape[0] == 0:
            if not math.isfinite(self.state.best_value):
                # Restart design done: its best value is the new reference
                self.state.best_value = self.train_y.max().item()
            self._fit_model()

    def _fit_model(self) -> None:
        """Refit the ARD weights when due, then the surrogate on the last D points."""
        if self._ard_weights is None or self._n_fits % self.ard_refit_period == 0:
            covar_module = SphericalLinearKernel(ard_num_dims=self.input_dim)
            covar_module.set_bounds(
                torch.stack([
                    torch.zeros(self.input_dim, dtype=self.dtype, device=self.device),
                    torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
                ])
            )
            model = SingleTaskGP(self.train_X, self.train_y, covar_module=covar_module)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            lengthscale = covar_module.lengthscale.detach().squeeze(0)
            # Normalize to unit geometric mean, as TuRBO does
            self._ard_weights = lengthscale / lengthscale.log().mean().exp()
        self._n_fits += 1

        lo, hi = self._search_box()
        window = slice(-self.input_dim, None)
        self._fit(
            self.train_X[window], self.train_y[window].squeeze(-1), centre=(lo + hi) / 2
        )

    def _search_box(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The trust region around the best point, clamped to [0,1]^d."""
        best = self.train_X[int(torch.argmax(self.train_y))]
        half = self._ard_weights * self.state.length / 2.0
        return (best - half).clamp(0.0, 1.0), (best + half).clamp(0.0, 1.0)

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "M D"]:
        """Suggest the next points, returning pending restart points first.

        Args:
            n: Number of points.
            seed: Random seed.

        Returns:
            Points to evaluate.
        """
        if self._restart_buffer.shape[0] > 0:
            points = self._restart_buffer[:n]
            self._restart_buffer = self._restart_buffer[n:]
            return points.cpu().numpy()

        # The failure tolerance depends on the batch size, as in TuRBO
        if self.state.batch_size != n:
            self.state.batch_size = n
            self.state.failure_tolerance = math.ceil(max(4.0 / n, self.input_dim / n))
        return super().ask(n, seed)
