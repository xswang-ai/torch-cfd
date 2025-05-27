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

# Modifications copyright (C) 2025 S.Cao
# ported Google's Jax-CFD functional template to PyTorch's tensor ops


import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

import torch_cfd.finite_differences as fdm

from torch_cfd import boundaries, grids

Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def safe_div(x, y, default_numerator=1):
    """Safe division of `Array`'s."""
    return x / torch.where(y != 0, y, default_numerator)


def van_leer_limiter(r):
    """Van-leer flux limiter."""
    return torch.where(r > 0, safe_div(2 * r, 1 + r), 0.0)


class Upwind(nn.Module):
    """Upwind interpolation of `c` to `offset` based on velocity field `v`.

    Interpolates values of `c` to `offset` in two steps:
    1) Identifies the axis along which `c` is interpolated. (must be single axis)
    2) For positive (negative) velocity along interpolation axis uses value from
       the previous (next) cell along that axis correspondingly.

    Args:
      c: quantitity to be interpolated.
      offset: offset to which `c` will be interpolated. Must have the same
        length as `c.offset` and differ in at most one entry.
      v: velocity field with offsets at faces of `c`. One of the components
        must have the same offset as `offset`.
      dt: size of the time step. Not used.

    Returns:
      A `GridVariable` that containins the values of `c` after interpolation to
      `offset`.

    Raises:
      InconsistentOffsetError: if `offset` and `c.offset` differ in more than one entry.
    """
    def __init__(self,
                 grid: Grid,
                 offset: Tuple[float, ...] = (0.5, 0.5),
                 offset_v: Tuple[Tuple[float, ...], ...] = ((1.0, 0.5), (0.5, 1.0))):
        super().__init__()
        self.grid = grid
        self.offset = offset
        self.offset_v = offset_v

    def forward(
        self,
        c: GridVariable,
        offset: Tuple[float, ...],
        v: GridVariableVector,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        if c.offset == offset:
            return c.data
        interpolation_axes = tuple(
            axis for axis, (cur, tgt) in enumerate(zip(c.offset, offset)) if cur != tgt
        )
        if len(interpolation_axes) != 1:
            raise ValueError(
                f"Offsets must differ in one entry, got {c.offset} vs {offset}"
            )
        (axis,) = interpolation_axes
        u = v[axis]
        
        offset_delta = self.offset_v[axis][axis] - self.offset[axis]
        # this should be the same with u.offset[axis] - c.offset[axis]
        if int(offset_delta) == offset_delta:
            return c.shift(int(offset_delta), axis).data
        floor = int(math.floor(offset_delta))
        ceil = int(math.ceil(offset_delta))
        c_floor = c.shift(floor, axis).data
        c_ceil = c.shift(ceil, axis).data
        return  torch.where(u.data > 0, c_floor, c_ceil)


class LaxWendroff(nn.Module):
    """Lax_Wendroff interpolation of `c` to `offset` based on velocity field `v`.

    Interpolates values of `c` to `offset` in two steps:
    1) Identifies the axis along which `c` is interpolated. (must be single axis)
    2) For positive (negative) velocity along interpolation axis uses value from
       the previous (next) cell along that axis plus a correction originating
       from expansion of the solution at the half step-size.

    This method is second order accurate with fixed coefficients and hence can't
    be monotonic due to Godunov's theorem.
    https://en.wikipedia.org/wiki/Godunov%27s_theorem

    Lax-Wendroff method can be used to form monotonic schemes when augmented with
    a flux limiter. See https://en.wikipedia.org/wiki/Flux_limiter

    Args:
      c: quantitity to be interpolated.
      offset: offset to which we will interpolate `c`. Must have the same
        length as `c.offset` and differ in at most one entry.
      v: velocity field with offsets at faces of `c`. One of the components must
        have the same offset as `offset`.
      dt: size of the time step. Not used.

    Returns:
      A `GridVariable` that containins the values of `c` after interpolation to
      `offset`.
    Raises:
      InconsistentOffsetError: if `offset` and `c.offset` differ in more than one entry.
    """
    def __init__(self, grid: Grid,
                 offset: Tuple[float, ...] = (0.5, 0.5),
                 offset_v: Tuple[Tuple[float, ...], ...] = ((1.0, 0.5), (0.5, 1.0))):
        super().__init__()
        self.grid = grid
        self.offset = offset
        self.offset_v = offset_v

    def forward(
        self,
        c: GridVariable,
        offset: Tuple[float, ...],
        v: GridVariableVector,
        dt: float = 1.0,
    ) -> torch.Tensor:
        if c.offset == offset:
            return c.data
        interpolation_axes = tuple(
            axis for axis, (cur, tgt) in enumerate(zip(c.offset, offset)) if cur != tgt
        )
        if len(interpolation_axes) != 1:
            raise ValueError(
                f"Offsets must differ in one entry, got {c.offset} vs {offset}"
            )
        (axis,) = interpolation_axes
        u = v[axis]
        offset_delta = self.offset_v[axis][axis] - self.offset[axis]
        # offset_delta = u.offset[axis] - c.offset[axis]
        floor = int(math.floor(offset_delta))
        ceil = int(math.ceil(offset_delta))
        # grid = grids.consistent_grid_arrays(c, u)
        courant = (dt / self.grid.step[axis]) * u.data
        c_floor = c.shift(floor, axis).data
        c_ceil = c.shift(ceil, axis).data
        pos = c_floor + 0.5 * (1 - courant) * (c_ceil - c_floor)
        neg = c_ceil - 0.5 * (1 + courant) * (c_ceil - c_floor)
        return torch.where(u.data > 0, pos, neg)


class AdvectAligned(nn.Module):
    """
    Computes fluxes and the associated advection for aligned `cs` and `v`.

    The values `cs` should consist of a single quantity `c` that has been
    interpolated to the offset of the components of `v`. The components of `v` and
    `cs` should be located at the faces of a single (possibly offset) grid cell.
    We compute the advection as the divergence of the flux on this control volume.

    The boundary condition on the flux is inherited from the scalar quantity `c`.

    A typical example in three dimensions would have

    ```
    cs[0].offset == v[0].offset == (1., .5, .5)
    cs[1].offset == v[1].offset == (.5, 1., .5)
    cs[2].offset == v[2].offset == (.5, .5, 1.)
    ```

    In this case, the returned advection term would have offset `(.5, .5, .5)`.
    """

    def __init__(
        self,
        grid: Grid,
        bcs_c: Tuple[boundaries.BoundaryConditions, ...],
        bcs_v: Tuple[boundaries.BoundaryConditions, ...],
        offsets: Tuple[Tuple[float, ...], ...] = ((1.5, 0.5), (0.5, 1.5)),
        **kwargs,
    ):
        """
        Initialize the AdvectAligned module.

        Args:
            grid: The computational grid
            cs_bc: Boundary conditions for the gradient of scalar field components
            v_bc: Boundary conditions for each velocity component
        """
        super().__init__()
        self.grid = grid
        self.bcs_c = bcs_c
        self.bcs_v = bcs_v
        self.offsets = offsets

        # Pre-compute whether we have periodic boundary conditions
        self.is_periodic = all(
            boundaries.is_bc_periodic_boundary_conditions(bc, i)
            for i, bc in enumerate(bcs_c)
        )

        # Pre-compute boundary condition functions for flux if not periodic
        if not self.is_periodic:
            self.flux_bcs = tuple(
                boundaries.get_advection_flux_bc_from_velocity_and_scalar_bc(
                    bcs_v[i], bcs_c[i], i
                )
                for i in range(len(bcs_v))
            )

    def forward(self, cs: GridVariableVector, v: GridVariableVector) -> GridVariable:
        """
        Compute advection of aligned cs and v.

        Args:
            cs: A sequence of GridVariables; a single value `c` that has been
                interpolated so that it is aligned with each component of `v`.
            v: A sequence of GridVariable describing a velocity field. Should be
               defined on the same Grid as cs.

        Returns:
            An GridVariable containing the time derivative of `c` due to advection by `v`.

        Raises:
            ValueError: `cs` and `v` have different numbers of components.
        """
        if len(cs) != len(v):
            raise ValueError(
                "`cs` and `v` must have the same length; "
                f"got {len(cs)} vs. {len(v)}."
            )

        # Compute flux: cu
        flux = GridVariableVector(tuple(c * u for c, u in zip(cs, v)))

        # Apply boundary conditions to flux if not periodic
        if not self.is_periodic:
            flux = GridVariableVector(
                tuple(bc.impose_bc(f) for f, bc in zip(flux, self.flux_bcs))
            )

        # Return negative divergence of flux
        return -fdm.divergence(flux)


class LinearInterpolation(nn.Module):
    """Multi-linear interpolation of `c` to `offset`.

    Args:
        c: quantitity to be interpolated.
        offset: offset to which we will interpolate `c`. Must have the same length
        as `c.offset`.
        v: velocity field. Not used.
        dt: size of the time step. Not used.

    Returns:
        An `GridArray` containing the values of `c` after linear interpolation
        to `offset`. The returned value will have offset equal to `offset`.
    """
    def __init__(
        self,
        grid: Grid,
        offset: Tuple[float, ...] = (0.5, 0.5),
        bc: Optional[boundaries.BoundaryConditions] = None,
        interpolant: Callable = fdm.linear,
        **kwargs,
    ):
        super().__init__()
        self.grid = grid
        self.offset = offset # this should be the same with the c.offset
        self.bc = bc
        self.interpolant = interpolant

    def forward(self, u: GridVariable, offset: Tuple[float, ...], **kwargs) -> GridVariable:
        return self.interpolant(u, offset, **kwargs)


class TVDInterpolation(nn.Module):
    """Combines low and high accuracy interpolators to get Total Variation Diminishing (TVD) Interpolator.

    Generates high accuracy interpolator by combining stable lwo accuracy `upwind`
    interpolation and high accuracy (but not guaranteed to be stable)
    `interpolation_fn` to obtain stable higher order method. This implementation
    follows the procedure outined in:
    http://www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2012/Chapter_4.pdf

    Args:
        interpolation_fn: higher order interpolation methods. Must follow the same
        interface as other interpolation methods (take `c`, `offset`, `grid`, `v`
        and `dt` arguments and return value of `c` at offset `offset`).
        limiter: flux limiter function that evaluates the portion of the correction
        (high_accuracy - low_accuracy) to add to low_accuracy solution based on
        the ratio of the consequtive gradients. Takes array as input and return
        array of weights. For more details see:
        https://en.wikipedia.org/wiki/Flux_limiter

    Returns:
        Interpolation method that uses a combination of high and low order methods
        to produce monotonic interpolation method.
    """
    def __init__(
        self,
        grid: Grid,
        limiter: Callable = van_leer_limiter,
        **offset_kwargs
    ):
        super().__init__()
        self.grid = grid
        self.low_interp = Upwind(grid=grid, 
                                 **offset_kwargs)
        self.high_interp = LaxWendroff(grid=grid,
                                       **offset_kwargs)
        self.limiter = limiter

    def forward(
        self,
        c: GridVariable,
        offset: Tuple[float, ...],
        v: GridVariableVector,
        dt: float,
    ) -> GridVariable:
        """Interpolate scalar field c to offset using Van Leer flux limiting."""
        for axis, axis_offset in enumerate(offset):
            interpolation_offset = tuple(
                [
                    c_offset if i != axis else axis_offset
                    for i, c_offset in enumerate(c.offset)
                ]
            )
            if interpolation_offset != c.offset:
                if interpolation_offset[axis] - c.offset[axis] != 0.5:
                    raise NotImplementedError(
                        "Only forward interpolation to control volume faces is supported."
                    )
                c_low = self.low_interp(c, interpolation_offset, v, dt)
                c_high = self.high_interp(c, interpolation_offset, v, dt)
                c_left = c.shift(-1, axis).data
                c_right = c.shift(1, axis).data
                c_next_right = c.shift(2, axis).data
                pos_r = safe_div(c - c_left, c_right - c)
                neg_r = safe_div(c_next_right - c_right, c_right - c)
                pos_phi = self.limiter(pos_r).data
                neg_phi = self.limiter(neg_r).data
                u = v[axis]
                phi = torch.where(u > 0, pos_phi, neg_phi)
                interpolated = c_low - (c_low - c_high) * phi
                c = GridVariable(interpolated.data, interpolation_offset, c.grid, c.bc)
        return c


class AdvectionVanLeer(nn.Module):
    """
    Base class for advection modules.

    Ported from Jax-CFD advect_general and _advect_aligned function

    Computes advection of a scalar quantity `c` by the velocity field `v`.

    This function follows the following procedure:

    1. Interpolate each component of `v` to the corresponding face of the
        control volume centered on `c`.
    2. Interpolate `c` to the same control volume faces.
    3. Compute the flux `cu` using the aligned values.
    4. Set the boundary condition on flux, which is inhereited from `c`.
    5. Return the negative divergence of the flux.

    Args:

    - velocity_interp: method for interpolating velocity field `v`.
    - flux_interp: method for interpolating the gradient of the scalar field `c`.
    dt: unused time-step.

    """

    def __init__(
        self,
        dt: float = 1.0,
        grid: Grid = None,
        offset_c: Tuple[float, ...] = (0.5, 0.5),
        offset_v: Tuple[Tuple[float, ...], ...] = ((1.0, 0.5), (0.5, 1.0)),
        bc_c: boundaries.BoundaryConditions = boundaries.periodic_boundary_conditions(ndim=2),
        bc_v: Tuple[boundaries.BoundaryConditions, ...] = (
            boundaries.periodic_boundary_conditions(ndim=2),
            boundaries.periodic_boundary_conditions(ndim=2),
        ),
        limiter: Callable = van_leer_limiter,
        **kwargs,
    ):
        super().__init__()
        self.dt = dt
        self.limiter = limiter
        self.grid = grid

        self.offset_c = offset_c
        self.offset_v = offset_v
        self.bc_c = bc_c
        self.bc_v = bc_v
        
        self.target_offsets = grids.control_volume_offsets(*offset_c)
        self.advect_aligned = AdvectAligned(grid=grid, bcs_c=(bc_c, bc_c), bcs_v=bc_v, offsets=self.target_offsets)
        self.tvd_interp = TVDInterpolation(grid=grid,
                                           offset=offset_c,
                                           offset_v=self.target_offsets,)
        self.velocity_interp = nn.ModuleList(
            LinearInterpolation(grid=grid, offset=offset, bc=bc)
            for offset, bc in zip(offset_v, bc_v)
        )

    def flux_interp(
        self,
        c: GridVariable,
        offset: Tuple[float, ...],
        v: GridVariableVector,
        dt: float,
    ) -> GridVariable:
        return self.tvd_interp(c, offset, v, dt)

    def forward(
        self,
        c: GridVariable,
        v: GridVariableVector,
        dt: float = 1.0,
    ):
        if not boundaries.has_all_periodic_boundary_conditions(c):
            raise NotImplementedError("Only periodic boundaries are implemented.")

        aligned_v = GridVariableVector(
            tuple(self.velocity_interp[i](u, offset)
                for i, (u, offset) in enumerate(
                    zip(v, self.target_offsets)
                )
            )
        )

        aligned_c = GridVariableVector(
            tuple(self.flux_interp(c, offset, aligned_v, dt)
                for offset in self.target_offsets))

        return self.advect_aligned(aligned_c, aligned_v)


class ConvectionVector(nn.Module):
    """Computes convection of a vector field `v` by the velocity field `u`.
    

    This function follows the following procedure:

      1. Interpolate each component of `u` to the corresponding face of the
         control volume centered on `v`.
      2. Interpolate `v` to the same control volume faces.
      3. Compute the flux `vu` using the aligned values.
      4. Set the boundary condition on flux, which is inhereited from `v`.
      5. Return the negative divergence of the flux.

    Notes:
      - In FVM of 2D NSE, v = u.
      - The velocity field `u` is assumed to be defined on a staggered grid
            (MAC grid) with the same offsets as `v`.
      - before the computation, the grid information is completely know, so one does not have to check dimensions or perform manipulation on-the-fly.

    Args:
      v: the quantity (also velocity) to be transported.
      u: a velocity field.
      offsets: the offsets of the velocity field (MAC grid), should be the same for both u and v
      bcs: boundary conditions for the velocity field to be convected. (u's boundary conditions do not matter)
      dt: unused time-step.

    Returns:
      The directional derivative `v` due to advection by `u`.
    """

    def __init__(
        self,
        grid: Grid,
        offsets: Tuple[Tuple[float, ...], ...] = ((1.0, 0.5), (0.5, 1.0)),
        bcs: Tuple[boundaries.BoundaryConditions, ...] = (
            boundaries.periodic_boundary_conditions(ndim=2),
            boundaries.periodic_boundary_conditions(ndim=2),
        ),
        limiter: Callable = van_leer_limiter,
        **kwargs,
    ):
        super().__init__()

        self.advect = nn.ModuleList(
            AdvectionVanLeer(
                grid=grid,
                offset_c=offset,
                offset_v=offsets,
                bc_c=bc,
                bc_v=bcs,
                limiter=limiter,
            )
            for bc, offset in zip(bcs, offsets)
        )

    def forward(
        self, v: GridVariableVector, u: GridVariableVector, dt: Optional[float] = None
    ) -> GridVariableVector:
        r"""
        Compute the convection of a vector field v with u.
        Basically computes (u \cdot \nabla) v  as
        the divergence of the flux on the control volume.

        Args:
            v: GridVariableVector to be advected
            u: GridVariableVector velocity field
            dt: Time step (overrides the instance dt if provided)

        Returns:
            GridVariable containing the advection result
        """

        return GridVariableVector(
            tuple(self.advect[i](c, u, dt) for i, c in enumerate(v))
        )
