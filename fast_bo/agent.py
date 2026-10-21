"""Fast BO for continuous inputs."""

import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from botorch.acquisition.analytic import _log_ei_helper
from jaxtyping import Float
from scipy.optimize import minimize


def project_onto_unit_sphere(
    x: Float[torch.Tensor, "... D"],
) -> Float[torch.Tensor, "... D+1"]:
    """Map points from R^d onto the unit sphere S^d by inverse stereographic projection.

    Args:
        x: Input points in R^d.

    Returns:
        Points on the unit sphere in R^(d+1).
    """
    x_sq_norm = x.square().sum(dim=-1, keepdim=True)
    return torch.cat([2 * x, x_sq_norm - 1.0], dim=-1).mul(1.0 / (1.0 + x_sq_norm))


def dinkelbach(
    a: Float[torch.Tensor, "B D"],
    c: Float[torch.Tensor, " B"],
    z_lo: Float[torch.Tensor, " D"],
    z_hi: Float[torch.Tensor, " D"],
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Float[torch.Tensor, "B D"]:
    """Maximize ``(2 a.z + c (|z|^2 - 1)) / (1 + |z|^2)`` over a box, row by row.

    This is a linear function of the spherical features P(z). Each Dinkelbach step
    maximizes ``numerator - lam * denominator``, which separates over coordinates.

    Args:
        a: Linear coefficients.
        c: Coefficients of ``|z|^2 - 1``.
        z_lo: Lower corner of the box (must contain 0).
        z_hi: Upper corner of the box.
        tol: Convergence tolerance on the ratio.
        max_iter: Maximum number of Dinkelbach steps.

    Returns:
        The maximizers.
    """

    def ratio(z: torch.Tensor) -> torch.Tensor:
        sq = z.square().sum(-1)
        return (2.0 * (a * z).sum(-1) + c * (sq - 1.0)) / (1.0 + sq)

    z = torch.zeros_like(a)
    lam = ratio(z)
    for _ in range(max_iter):
        coef = (c - lam).unsqueeze(-1)
        # Concave: clip the stationary point. Otherwise: take the better endpoint.
        concave = coef < 0
        interior = (a / torch.where(concave, -coef, 1.0)).clamp(z_lo, z_hi)
        upper = coef * z_hi.square() + 2.0 * a * z_hi
        lower = coef * z_lo.square() + 2.0 * a * z_lo
        z = torch.where(concave, interior, torch.where(upper >= lower, z_hi, z_lo))
        new_lam = ratio(z)
        if bool(((new_lam - lam).abs() < tol).all()):
            break
        lam = new_lam
    return z


class FastLinearBO:
    """Fast BO on [0,1]^d (maximization).

    The surrogate is Bayesian linear regression on the spherical features
    ``u(x) = [sqrt(b1) P(z), sqrt(b0)]``, with ``z = (x - 0.5) / scale`` and
    ``b0 + b1 = 1``. This is Linear BO's kernel with a single lengthscale. The
    lengthscale is picked from a small grid and ``(b1, noise)`` are fit by Nelder-Mead
    on the closed-form marginal likelihood.

    TS maximizes a posterior sample exactly with Dinkelbach's method. For UCB and EI we
    linearize the posterior std at the current point, solve the linear problem with
    Dinkelbach and repeat (minorize-maximize), from several random starts.

    Args:
        input_dim: Input dimension.
        acquisition: "ucb", "ei" or "ts".
        dtype: Torch dtype.
        device: Torch device.
    """

    beta = 2.0  # UCB exploration weight
    n_starts = 10  # MM starts for UCB and EI
    mm_steps = 10  # max MM steps per start
    n_grid = 10  # lengthscale grid: scale^2 = f * d / 4, f geometric on [1e-3, 1]
    refresh_period = 10  # redo the lengthscale grid search every n fits
    noise_floor = 1e-8

    def __init__(
        self,
        input_dim: int,
        acquisition: str = "ucb",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        if acquisition not in {"ucb", "ei", "ts"}:
            msg = f"Unknown acquisition: {acquisition!r}"
            raise ValueError(msg)
        self.acquisition = acquisition
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim

        self.train_X = torch.empty(0, self.input_dim, dtype=dtype, device=device)
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.fraction = None  # current lengthscale (as a fraction, see n_grid)
        self._fits_since_refresh = 0
        self._pool = None

    def _fit_model(self) -> None:
        """Fit the surrogate on all data, centered at 0.5."""
        self._fit(self.train_X, self.train_y.squeeze(-1), centre=0.5)

    def _fit(
        self,
        x: Float[torch.Tensor, "N D"],
        y: Float[torch.Tensor, " N"],
        centre: float | Float[torch.Tensor, " D"],
    ) -> None:
        """Fit the hyperparameters and cache what the posterior needs."""
        self.y_mean, self.y_std = y.mean(), y.std().clamp_min(1e-12)
        y = (y - self.y_mean) / self.y_std
        self.centre = centre

        # Search the lengthscale grid every `refresh_period` fits, otherwise reuse the
        # last one. Threads only pay off for larger eigendecompositions.
        if self.fraction is None or self._fits_since_refresh + 1 >= self.refresh_period:
            grid = np.geomspace(1e-3, 1.0, self.n_grid)
            parallel = min(x.shape) >= 128
            fits = self._map(lambda f: self._fit_at(x, y, f), grid, parallel=parallel)
            best = int(np.argmin([fit[0] for fit in fits]))
            self.fraction, fit = float(grid[best]), fits[best]
            self._fits_since_refresh = 0
        else:
            fit = self._fit_at(x, y, self.fraction)
            self._fits_since_refresh += 1
        _, self.b1, self.noise, self.c, self.scale, proj, vals, vecs = fit

        # K = b1 P P^T + b0 11^T + noise I. Invert M = b1 P P^T + noise I in the
        # eigenbasis from the fit, and handle b0 11^T as a rank-one update.
        if proj.shape[0] > proj.shape[1]:
            coef = (self.b1 / (self.b1 * vals + self.noise)).unsqueeze(-1)
            self._m_inv = lambda r: (
                (r - proj @ (vecs @ (coef * (vecs.mT @ (proj.mT @ r))))) / self.noise
            )
        else:
            diag = (self.b1 * vals + self.noise).unsqueeze(-1)
            self._m_inv = lambda r: vecs @ ((vecs.mT @ r) / diag)
        b0 = 1.0 - self.b1
        ones = torch.ones_like(y).unsqueeze(-1)
        self._m_inv_one = self._m_inv(ones)
        self._rank_one = b0 / (1.0 + b0 * float(ones.mT @ self._m_inv_one))

        self._train_u = self._features(proj=proj)
        self._residual = (y - self.c).unsqueeze(-1)
        self._mean_weights = (self._train_u.mT @ self._kinv(self._residual)).squeeze(-1)

    def _fit_at(self, x: torch.Tensor, y: torch.Tensor, fraction: float) -> tuple:
        """Fit ``(b1, noise, c)`` at one lengthscale. Returns the NLL and the fit."""
        scale = math.sqrt(fraction * self.input_dim / 4.0)
        proj = project_onto_unit_sphere((x - self.centre) / scale)
        n, m = proj.shape
        ones = torch.ones_like(y)

        # Eigendecompose P P^T, or P^T P if that is smaller, and express 1 and y in the
        # eigenbasis of P P^T. In the second case the null space becomes one extra
        # entry with eigenvalue 0 and multiplicity n - rank.
        if n <= m:
            vals, vecs = torch.linalg.eigh(proj @ proj.mT)
            vals = vals.clamp_min(0.0)
            w = vals.cpu().numpy()
            one = (vecs.mT @ ones).cpu().numpy()
            tgt = (vecs.mT @ y).cpu().numpy()
            one_one, one_y, y_y, mult = one * one, one * tgt, tgt * tgt, np.ones_like(w)
        else:
            vals, vecs = torch.linalg.eigh(proj.mT @ proj)
            vals = vals.clamp_min(0.0)
            lam = vals.cpu().numpy()
            one = (vecs.mT @ (proj.mT @ ones)).cpu().numpy()
            tgt = (vecs.mT @ (proj.mT @ y)).cpu().numpy()
            keep = lam > 1e-10 * lam.max()
            w = lam[keep]
            one_one, one_y = one[keep] ** 2 / w, one[keep] * tgt[keep] / w
            y_y = tgt[keep] ** 2 / w
            w = np.append(w, 0.0)
            one_one = np.append(one_one, n - one_one.sum())
            one_y = np.append(one_y, float(y.sum()) - one_y.sum())
            y_y = np.append(y_y, float(y @ y) - y_y.sum())
            mult = np.append(np.ones(len(w) - 1), n - keep.sum())

        # With M = b1 P P^T + noise I, g = 1'M^-1 1, t = 1'M^-1 y and q = y'M^-1 y,
        # the profiled mean is c = t / g and
        #   -log p(y) = (q - t^2 / g + log det M + log(1 + b0 g)) / 2 + const.
        # Nelder-Mead starts at b1 = 0.5, noise = 0.1 and stops after 80 evaluations.
        def params(p: np.ndarray) -> tuple[float, float]:
            b1 = 0.5 * (1.0 + math.tanh(0.5 * p[0]))  # sigmoid
            return b1, max(math.exp(min(p[1], 700.0)), self.noise_floor)

        def nll(p: np.ndarray) -> tuple[float, float]:
            b1, noise = params(p)
            diag = b1 * w + noise
            inv = 1.0 / diag
            g, t, q = float(one_one @ inv), float(one_y @ inv), float(y_y @ inv)
            val = (
                0.5 * (q - t * t / g)
                + 0.5 * float(mult @ np.log(diag))
                + 0.5 * math.log1p((1.0 - b1) * g)
            )
            return (val if math.isfinite(val) else math.inf), t / g

        res = minimize(
            lambda p: nll(p)[0],
            x0=[0.0, math.log(0.1)],
            method="Nelder-Mead",
            options={"maxfev": 80, "xatol": 1e-3, "fatol": 1e-4},
        )
        b1, noise = params(res.x)
        return res.fun, b1, noise, nll(res.x)[1], scale, proj, vals, vecs

    def _map(self, fn: Callable, grid: np.ndarray, *, parallel: bool) -> list:
        """Evaluate ``fn`` over the grid, on a thread pool if ``parallel``."""
        if not parallel:
            return [fn(float(f)) for f in grid]
        if self._pool is None:
            try:
                n_cpus = len(os.sched_getaffinity(0))
            except AttributeError:  # macOS
                n_cpus = os.cpu_count() or 1
            self._pool = ThreadPoolExecutor(max_workers=min(len(grid), n_cpus))
        # One BLAS thread per worker
        n_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            return list(self._pool.map(lambda f: fn(float(f)), grid))
        finally:
            torch.set_num_threads(n_threads)

    def _features(
        self, z: torch.Tensor | None = None, proj: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Features ``[sqrt(b1) P(z), sqrt(b0)]`` from ``z`` or a precomputed P(z)."""
        proj = project_onto_unit_sphere(z) if proj is None else proj
        const = proj.new_full((*proj.shape[:-1], 1), math.sqrt(1.0 - self.b1))
        return torch.cat([math.sqrt(self.b1) * proj, const], dim=-1)

    def _kinv(self, rhs: torch.Tensor) -> torch.Tensor:
        """Compute ``K^{-1} rhs`` for ``rhs`` of shape ``(N, B)``."""
        m_inv_one = self._m_inv_one
        return self._m_inv(rhs) - self._rank_one * m_inv_one @ (m_inv_one.mT @ rhs)

    def _sigma(self, u: torch.Tensor) -> torch.Tensor:
        """Posterior weight covariance times each row of ``u``."""
        return u - (self._train_u.mT @ self._kinv(self._train_u @ u.mT)).mT

    def tell_initial_data(
        self, x: Float[np.ndarray, "N D"], y: Float[np.ndarray, " N"], seed: int
    ) -> None:
        """Set the initial data and fit.

        Args:
            x: Initial points.
            y: Observed values.
            seed: Unused.
        """
        self.train_X = torch.tensor(x, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            y, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self._fit_model()

    def tell(
        self, x: Float[np.ndarray, "N D"], y: Float[np.ndarray, " N"], seed: int
    ) -> None:
        """Add new data and refit.

        Args:
            x: New points.
            y: Observed values.
            seed: Unused here, but used by the trust-region subclass.
        """
        new_X = torch.tensor(x, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self._fit_model()

    def _search_box(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Corners of the search region, here the whole unit cube."""
        lo = torch.zeros(self.input_dim, dtype=self.dtype, device=self.device)
        return lo, lo + 1.0

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "M D"]:
        """Suggest the next points. UCB and EI return one point, TS returns ``n``.

        Args:
            n: Number of points.
            seed: Random seed.

        Returns:
            Points to evaluate.

        Raises:
            RuntimeError: If called before `tell_initial_data`.
        """
        if self.fraction is None:
            msg = "Model has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        lo, hi = self._search_box()
        z_lo, z_hi = (lo - self.centre) / self.scale, (hi - self.centre) / self.scale
        generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            if self.acquisition == "ts":
                z = self._thompson(z_lo, z_hi, n, generator)
            else:
                z = self._maximize_ucb_ei(z_lo, z_hi, generator).unsqueeze(0)
        x = (self.centre + self.scale * z).clamp(0.0, 1.0)
        return x.cpu().numpy()

    def _thompson(
        self, z_lo: torch.Tensor, z_hi: torch.Tensor, n: int, generator: torch.Generator
    ) -> torch.Tensor:
        """Draw ``n`` posterior samples (Matheron's rule) and maximize each exactly."""
        n_data, n_features = self._train_u.shape
        # The generator lives on the CPU
        prior = torch.randn(n_features, n, generator=generator, dtype=self.dtype)
        noise = math.sqrt(self.noise) * torch.randn(
            n_data, n, generator=generator, dtype=self.dtype
        )
        prior, noise = prior.to(self.device), noise.to(self.device)
        rhs = self._residual - self._train_u @ prior - noise
        w = (prior + self._train_u.mT @ self._kinv(rhs)).mT
        return dinkelbach(w[:, : self.input_dim], w[:, self.input_dim], z_lo, z_hi)

    def _maximize_ucb_ei(
        self, z_lo: torch.Tensor, z_hi: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Maximize UCB or EI by minorize-maximize from random starts (batched)."""
        d, theta = self.input_dim, self._mean_weights
        best = float((self.train_y.max() - self.y_mean) / self.y_std)

        z = z_lo + (z_hi - z_lo) * torch.rand(
            self.n_starts, d, generator=generator, dtype=self.dtype
        ).to(self.device)
        u = self._features(z)
        for _ in range(self.mm_steps):
            sigma_u = self._sigma(u)
            mean, std = self.c + u @ theta, (u * sigma_u).sum(-1).clamp_min(0.0).sqrt()
            # Weight on the std: beta for UCB, phi(gamma) / Phi(gamma) for EI
            if self.acquisition == "ucb":
                lam = torch.full_like(std, self.beta)
            else:
                gamma = (mean - best) / std.clamp_min(1e-8)
                log_phi = -0.5 * gamma.square() - 0.5 * math.log(2 * math.pi)
                lam = torch.exp(log_phi - torch.special.log_ndtr(gamma))
            coef = torch.where(std >= 1e-8, lam / std.clamp_min(1e-8), 0.0)
            w = theta + coef.unsqueeze(-1) * sigma_u
            proposal = dinkelbach(w[:, :d], w[:, d], z_lo, z_hi)
            new_u = self._features(proposal)
            # Reject steps that decrease the linear minorant
            accept = ((w * (new_u - u)).sum(-1) >= 0).unsqueeze(-1)
            new_z = torch.where(accept, proposal, z)
            converged = bool((torch.linalg.vector_norm(new_z - z, dim=-1) < 1e-6).all())
            z, u = new_z, torch.where(accept, new_u, u)
            if converged:
                break

        mean = self.c + u @ theta
        std = (u * self._sigma(u)).sum(-1).clamp_min(0.0).sqrt()
        if self.acquisition == "ucb":
            value = mean + self.beta * std
        else:
            # Use log EI, plain EI underflows for very negative gamma
            gamma = (mean - best) / std.clamp_min(1e-12)
            value = std.clamp_min(1e-12).log() + _log_ei_helper(gamma)
            value = torch.where(std < 1e-12, (mean - best).clamp_min(0.0).log(), value)
        # Skip points we have already evaluated (unless all of them were)
        x = (self.centre + self.scale * z).clamp(0.0, 1.0)
        seen = (x[:, None] == self.train_X[None]).all(-1).any(-1)
        if not bool(seen.all()):
            value = value.masked_fill(seen, -math.inf)
        return z[int(torch.argmax(value))]
