# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (C) 2024 S.Cao
# ported Google's Jax-CFD functional template to PyTorch's tensor ops

"""Prepare initial conditions for simulations."""
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.fft as fft
import torch.nn as nn

from torch_cfd import boundaries, grids, pressure

Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions
ScalarFn = Callable[..., torch.Tensor]
ScalarFns = Sequence[ScalarFn]
VelocityFn = Tuple[ScalarFn, ...]
VelocityFns = Sequence[VelocityFn]


def wrap_velocities(
    v: Sequence[torch.Tensor],
    grid: Grid,
    bcs: Sequence[BoundaryConditions],
    device: Optional[torch.device] = None,
) -> GridVariableVector:
    """Wrap velocity arrays for input into simulations."""
    device = grid.device if device is None else device
    velocity = tuple(
        GridVariable(u, offset, grid) for u, offset in zip(v, grid.cell_faces)
    )
    return GridVariableVector(
        tuple(bc.impose_bc(u) for u, bc in zip(velocity, bcs))
    ).to(device)


def wrap_vorticity(
    w: torch.Tensor,
    grid: Grid,
    bc: BoundaryConditions,
    device: Optional[torch.device] = None,
) -> GridVariable:
    """Wrap vorticity arrays for input into simulations."""
    device = grid.device if device is None else device
    vorticity = GridVariable(w, grid.cell_center, grid)
    return bc.impose_bc(vorticity).to(device)


def log_normal_density(k, mode: float, variance=0.25):
    """
    Unscaled PDF for a log normal given `mode` and log variance 1.
    """
    mean = math.log(mode) + variance
    logk = torch.log(k)
    return torch.exp(-((mean - logk) ** 2) / 2 / variance - logk)


def McWilliams_density(k: torch.Tensor, mode: float, tau: float = 1.0):
    r"""Implements the McWilliams spectral density function.
    |\psi|^2 \sim k^{-1}(tau^2 + (k/k_0)^4)^{-1}
    k_0 is a prescribed wavenumber that the energy peaks.
    tau flattens the spectrum density at low wavenumbers to be bigger.

    Refs:
      McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow.
    """
    return (k * (tau**2 + (k / mode) ** 4)) ** (-1)


def _angular_frequency_magnitude(grid: grids.Grid) -> torch.Tensor:
    frequencies = [
        2 * torch.pi * fft.fftfreq(size, step)
        for size, step in zip(grid.shape, grid.step)
    ]
    freq_vector = torch.stack(torch.meshgrid(*frequencies, indexing="ij"), dim=0)
    return torch.linalg.norm(freq_vector, dim=0)


def grf_spectral_filter(
    spectral_density: Callable[[torch.Tensor], torch.Tensor],
    v: Union[torch.Tensor, GridVariable],
    grid: Grid,
) -> torch.Tensor:
    """Generate a Gaussian random field with white noise to match a prescribed spectral density."""
    k = _angular_frequency_magnitude(grid)
    filters = torch.where(k > 0, spectral_density(k), 0.0).to(v.device)
    # The output signal can safely be assumed to be real if our input signal was
    # real, because our spectral density only depends on norm(k).
    return fft.ifftn(fft.fftn(v) * filters).real


def streamfunc_normalize(k, psi):
    nx, ny = psi.shape
    psih = fft.fft2(psi)
    uh_mag = k * psih
    kinetic_energy = (2 * uh_mag.abs() ** 2 / (nx * ny) ** 2).sum()
    return psi / kinetic_energy.sqrt()


def project_and_normalize(
    v: GridVariableVector,
    maximum_velocity: float = 1,
    projection: Optional[nn.Module] = None,
) -> GridVariableVector:
    grid = grids.consistent_grid_arrays(*v)
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
    if projection is None:
        projection = pressure.PressureProjection(grid, pressure_bc).to(v.device)
    v, _ = projection(v)
    vmax = torch.linalg.norm(torch.stack([u.data for u in v]), dim=0).max()
    v = GridVariableVector(
        tuple(
            GridVariable(maximum_velocity * u.data / vmax, u.offset, u.grid, u.bc)
            for u in v
        )
    )
    return v


def filtered_velocity_field(
    grid: Grid,
    maximum_velocity: float = 1,
    peak_wavenumber: float = 3,
    iterations: int = 3,
    random_state: int = 0,
    batch_size: int = 1,
    spectral_density: Callable[
        [torch.Tensor, float], torch.Tensor
    ] = log_normal_density,
    device: torch.device = torch.device("cpu"),
) -> GridVariableVector:
    """Create divergence-free velocity fields with appropriate spectral filtering.

    Args:
      rng_key: key for seeding the random initial velocity field.
      grid: the grid on which the velocity field is defined.
      maximum_velocity: the maximum speed in the velocity field.
      peak_wavenumber: the velocity field will be filtered so that the largest
        magnitudes are associated with this wavenumber.
      iterations: the number of repeated pressure projection and normalization
        iterations to apply.
    Returns:
      A divergence free velocity field with the given maximum velocity. Associates
      periodic boundary conditions with the velocity field components.
    """

    # Log normal distribution peaked at `peak_wavenumber`. Note that we have to
    # divide by `k ** (ndim - 1)` to account for the volume of the
    # `ndim - 1`-sphere of values with wavenumber `k`.
    _spectral_density = lambda k: spectral_density(k, peak_wavenumber) / k ** (
        grid.ndim - 1
    )
    result = []

    for k in range(batch_size):
        random_states = [random_state + i + k * batch_size for i in range(grid.ndim)]
        rng = torch.Generator(device=device)
        velocity_components = []
        boundary_conditions = []
        for s in random_states:
            rng.manual_seed(s)
            noise = torch.randn(grid.shape, generator=rng, device=device)
            velocity_components.append(grf_spectral_filter(_spectral_density, noise, grid))
            boundary_conditions.append(
                boundaries.periodic_boundary_conditions(grid.ndim)
            )
        velocity = wrap_velocities(
            velocity_components, grid, boundary_conditions, device=device
        )
        for _ in range(iterations):
            velocity = project_and_normalize(velocity, maximum_velocity)
        result.append(velocity)
        # Original comment by Jax-cfd: Due to numerical precision issues, we repeatedly normalize and project the
        # velocity field. This ensures that it is divergence-free and achieves the specified maximum velocity.
        # Porting notes: using fp64, projecting once is enough
        # velocity is ((n, n), (n, n)) GridVariableVector

    return grids.stack_gridvariable_vectors(*result)


def filtered_vorticity_field(
    grid: Grid,
    peak_wavenumber: float = 3,
    random_state: int = 0,
    batch_size: int = 1,
    spectral_density: Callable[
        [torch.Tensor, float], torch.Tensor
    ] = McWilliams_density,
    device: torch.device = torch.device("cpu"),
) -> GridVariable:
    """Create vorticity field with a spectral filtering
    using the McWilliams power spectrum density function.

    Args:
      rng_key: key for seeding the random initial vorticity field.
      grid: the grid on which the vorticity field is defined.
      peak_wavenumber: the velocity field will be filtered so that the largest
        magnitudes are associated with this wavenumber.

    Returns:
      A vorticity field with periodic boundary condition.
    """
    _spectral_density = lambda k: spectral_density(k, peak_wavenumber)

    rng = torch.Generator()
    result = []

    for k in range(batch_size):
        random_state = random_state + k
        rng.manual_seed(random_state)
        noise = torch.randn(grid.shape, generator=rng)
        k = _angular_frequency_magnitude(grid)
        psi = grf_spectral_filter(_spectral_density, noise, grid)
        psi = streamfunc_normalize(k, psi)
        vorticity = fft.ifftn(fft.fftn(psi) * k**2).real
        boundary_condition = boundaries.periodic_boundary_conditions(grid.ndim)
        vorticity = wrap_vorticity(vorticity, grid, boundary_condition, device=device)
        result.append(vorticity)

    return grids.stack_gridvariables(*result)


def velocity_field(
    velocity_fns: Union[VelocityFns, VelocityFn],
    grid: Grid,
    velocity_bc: Optional[Sequence[BoundaryConditions]] = None,
    batch_size: int = 1,
    noise: float = 0.0,
    random_state: int = 0,
    peak_wavenumber: float = 3,
    spectral_density: Optional[
        Callable[[torch.Tensor, float], torch.Tensor]
    ] = log_normal_density,
    iterations: int = 0,
    maximum_velocity: float = 1,
    projection: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> GridVariableVector:
    """Given velocity functions on arrays, returns the velocity field on the grid, with potentially added noise.

    Typical usage example:
      grid = grids.Grid((128, 128))
      x_velocity_fn = lambda x, y: torch.sin(x) * torch.cos(y)
      y_velocity_fn = lambda x, y: torch.zeros_like(x)
      v0 = velocity_field((x_velocity_fn, y_velocity_fn), grid)

    Args:
      velocity_fns: functions for computing each velocity component. These should
        takes the args (x, y, ...) and return an array of the same shape.
      grid: the grid on which the velocity field is defined.
      velocity_bc: the boundary conditions to associate with each velocity
        component. If unspecified, uses periodic boundary conditions.
      batch_size: number of velocity fields to generate. If there is only one sample, there will be no noise added. For multiple samples, noise will be added starting from the second sample if noise > 0.
      noise: if specified, the standard deviation of the Gaussian noise to add
        to the velocity field (in spatial domain).
      maximum_velocity: the maximum speed in the velocity field.
      projection: method used to solve pressure projection.
      iterations: if specified, the number of iterations of applied projection
        onto an incompressible velocity field.

    Returns:
      Velocity components defined with expected offsets on the grid.
    """
    if velocity_bc is None:
        velocity_bc = (boundaries.periodic_boundary_conditions(grid.ndim),) * grid.ndim
    if spectral_density:
        _spectral_density = lambda k: spectral_density(k, peak_wavenumber) / k ** (
            grid.ndim - 1
        )

    # Handle the case where velocity_fns is a single function vs a sequence
    all_velocity_fns: List[Any]
    if isinstance(velocity_fns, tuple) and len(velocity_fns) == grid.ndim:
        # Case 1: Single set of velocity functions to be replicated for each batch
        all_velocity_fns = [velocity_fns for _ in range(batch_size)]
        # Case 2: Different velocity functions for each batch sample
    elif len(velocity_fns) == batch_size:
        all_velocity_fns = list(velocity_fns)
    else:
        raise TypeError(
            f"velocity_fns must be given either as a sequence to produce batched velocities, or as a {grid.ndim}D tuple of functions."
        )

    result = []

    for k in range(batch_size):
        v = [
            grid.eval_on_mesh(v_fn, offset).data
            for v_fn, offset in zip(all_velocity_fns[k], grid.cell_faces)
        ]
        if noise > 0 and k >= 1:
            random_states = [
                random_state + i + k * batch_size for i in range(grid.ndim)
            ]
            rng = torch.Generator(device=device)
            perturbation = []
            for s in random_states:
                rng.manual_seed(s)
                _noise = torch.randn(grid.shape, generator=rng, device=device)
                _perturbation = grf_spectral_filter(_spectral_density, _noise, grid)
                _perturbation /= _perturbation.abs().max()  # normalize
                _perturbation *= maximum_velocity * noise
                perturbation.append(_perturbation)
            v = [u + e for u, e in zip(v, perturbation)]

        velocity = wrap_velocities(v, grid, velocity_bc, device=device)

        if iterations > 0:
            for _ in range(iterations):
                velocity = project_and_normalize(
                    velocity, maximum_velocity=maximum_velocity, projection=projection
                )
        result.append(velocity)

    return grids.stack_gridvariable_vectors(*result)

def vorticity_field(
    vorticity_fns: Union[ScalarFn, ScalarFns],
    grid: Grid,
    vorticity_bc: Optional[BoundaryConditions] = None,
    batch_size: int = 1,
    noise: float = 0.0,
    random_state: int = 0,
    peak_wavenumber: float = 3,
    spectral_density: Optional[
        Callable[[torch.Tensor, float], torch.Tensor]
    ] = McWilliams_density,
    device: Optional[torch.device] = None,
) -> GridVariable:
    """Given vorticity functions on arrays, returns the vorticity field on the grid, with potentially added noise.

    Typical usage example:
      grid = grids.Grid((128, 128))
      vorticity_fn = lambda x, y: torch.sin(x) * torch.cos(y)
      w0 = vorticity_field(vorticity_fn, grid)

    Args:
      vorticity_fns: function(s) for computing vorticity. For a single function,
        it should take args (x, y, ...) and return an array of the same shape.
        For multiple functions (batched), provide a sequence of such functions.
      grid: the grid on which the vorticity field is defined.
      vorticity_bc: the boundary conditions to associate with the vorticity
        field. If unspecified, uses periodic boundary conditions.
      batch_size: number of vorticity fields to generate. If there is only one 
        sample, there will be no noise added. For multiple samples, noise will 
        be added starting from the second sample if noise > 0.
      noise: if specified, the standard deviation of the Gaussian noise to add
        to the vorticity field (in spatial domain).
      peak_wavenumber: the wavenumber at which the spectral noise peaks.
      spectral_density: spectral density function for noise generation.
      device: device to place the tensors on.

    Returns:
      Vorticity field defined on the grid with specified boundary conditions.
    """
    if vorticity_bc is None:
        vorticity_bc = boundaries.periodic_boundary_conditions(grid.ndim)
    
    if spectral_density:
        _spectral_density = lambda k: spectral_density(k, peak_wavenumber)

    # Handle the case where vorticity_fns is a single function vs a sequence
    all_vorticity_fns: List[Any]
    if callable(vorticity_fns):
        # Case 1: Single vorticity function to be replicated for each batch
        all_vorticity_fns = [vorticity_fns for _ in range(batch_size)]
    elif len(vorticity_fns) == batch_size:
        # Case 2: Different vorticity functions for each batch sample
        all_vorticity_fns = list(vorticity_fns)
    else:
        raise TypeError(
            f"vorticity_fns must be given either as a single function or as a sequence of {batch_size} functions for batched vorticities."
        )

    result = []

    for k in range(batch_size):
        # Evaluate the vorticity function on the grid
        w = grid.eval_on_mesh(all_vorticity_fns[k], grid.cell_center).data
        
        # Add noise to samples after the first one if requested
        if noise > 0 and k >= 1:
            rng = torch.Generator(device=device)
            rng.manual_seed(random_state + k)
            _noise = torch.randn(grid.shape, generator=rng, device=device)
            _perturbation = grf_spectral_filter(_spectral_density, _noise, grid)
            _perturbation /= _perturbation.abs().max()  # normalize
            _perturbation *= noise
            w = w + _perturbation

        # Wrap the vorticity with boundary conditions
        vorticity = wrap_vorticity(w, grid, vorticity_bc, device=device)
        result.append(vorticity)

    return grids.stack_gridvariables(*result)
