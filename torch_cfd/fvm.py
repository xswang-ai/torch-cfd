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
"""Finite volume methods on MAC grids with pressure projection."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import torch_cfd.finite_differences as fdm
from torch_cfd import advection, boundaries, forcings, grids, solvers

default = advection.default
Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = boundaries.BoundaryConditions
ForcingFn = forcings.ForcingFn
Solver = solvers.SolverBase


def wrap_field_same_bcs(v, field_ref):
    return GridVariableVector(
        tuple(
            GridVariable(a.data, a.offset, a.grid, w.bc) for a, w in zip(v, field_ref)
        )
    )


class ProjectionExplicitODE(nn.Module):
    r"""Navier-Stokes equation in 2D with explicit stepping and a pressure projection (discrete Helmholtz decomposition by modding the gradient of a Laplacian inverse of the extra divergence).

    \partial u/ \partial t = explicit_terms(u)
    u <- pressure_projection(u)
    """

    def explicit_terms(self, *args, **kwargs) -> GridVariableVector:
        """
        Explicit forcing term as du/dt.
        """
        raise NotImplementedError

    def pressure_projection(
        self, *args, **kwargs
    ) -> Tuple[GridVariableVector, GridVariable]:
        """Pressure projection step."""
        raise NotImplementedError

    def forward(self, u: GridVariableVector, dt: float) -> GridVariableVector:
        """Perform one time step.

        Args:
            u: Initial state (velocity field)
            dt: Time step size

        Returns:
            Updated velocity field after one time step
        """
        raise NotImplementedError


class RKStepper(nn.Module):
    """Base class for Explicit Runge-Kutta stepper.

    Input:
        tableau: Butcher tableau (a, b) for the Runge-Kutta method as a dictionary
        method: String name of built-in RK method if tableau not provided

    Examples:
        stepper = RKStepper.from_method("classic_rk4", ...)
    """

    _METHOD_MAP = {
        "forward_euler": {"a": [], "b": [1.0]},
        "midpoint": {"a": [[1 / 2]], "b": [0, 1.0]},
        "heun_rk2": {"a": [[1.0]], "b": [1 / 2, 1 / 2]},
        "classic_rk4": {
            "a": [[1 / 2], [0.0, 1 / 2], [0.0, 0.0, 1.0]],
            "b": [1 / 6, 1 / 3, 1 / 3, 1 / 6],
        },
    }

    def __init__(
        self,
        tableau: Optional[Dict[str, List]] = None,
        method: Optional[str] = "forward_euler",
        dtype: Optional[torch.dtype] = torch.float32,
        requires_grad=False,
        **kwargs,
    ):
        super().__init__()

        self._method = None
        self.dtype = dtype
        self.requires_grad = requires_grad

        # Set the tableau first directly, either directly or from method name
        if tableau is not None:
            self.tableau = tableau
        elif method is not None:
            self.method = method
        self._set_params()

    @property
    def method(self):
        """Get the current Runge-Kutta method name."""
        return self._method

    @method.setter
    def method(self, name: str):
        """Set the tableau based on the method name."""
        if name not in self._METHOD_MAP:
            raise ValueError(f"Unknown RK method: {name}")
        self._method = name
        self._tableau = self._METHOD_MAP[name]

    @property
    def tableau(self):
        """Get the current tableau."""
        return self._tableau

    @tableau.setter
    def tableau(self, tab: Dict[str, List]):
        """Set the tableau directly."""
        self._tableau = tab
        self._method = None  # Clear method name when setting tableau directly

    def _set_params(self):
        """Set the parameters of the Butcher tableau."""
        try:
            a, b = self._tableau["a"], self._tableau["b"]
            if a.__len__() + 1 != b.__len__():
                raise ValueError("Inconsistent Butcher tableau: len(a) + 1 != len(b)")
            self.params = nn.ParameterDict()
            self.params["a"] = nn.ParameterList()
            for a_ in a:
                self.params["a"].append(
                    nn.Parameter(
                        torch.tensor(
                            a_, dtype=self.dtype, requires_grad=self.requires_grad
                        )
                    )
                )
            self.params["b"] = nn.Parameter(
                torch.tensor(b, dtype=self.dtype, requires_grad=self.requires_grad)
            )
        except KeyError as e:
            print(f"{e}: Either `tableau` or `method` must be given.")

    @classmethod
    def from_tableau(
        cls,
        tableau: Dict[str, List],
        requires_grad: bool = False,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        """Factory method to create an RKStepper from a Butcher tableau."""
        return cls(tableau=tableau, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def from_method(
        cls, method: str = "forward_euler", requires_grad: bool = False, **kwargs
    ):
        """Factory method to create an RKStepper by name."""
        return cls(method=method, requires_grad=requires_grad, **kwargs)

    def forward(
        self,
        u0: GridVariableVector,
        dt: float,
        equation: ProjectionExplicitODE,
        t: float = 0.0,
    ) -> Tuple[GridVariableVector, GridVariable]:
        """Perform one time step.

        Args:
            u0: Initial state (velocity field)
            dt: Time step size
            equation: The ODE to solve

        Returns:
            Updated velocity field after one time step

        Port note:
        - In Jax-CFD, dvdt is wrapped with the same bc with v,
          which does not work for inhomogeneous boundary condition.
          see explicit_terms_with_same_bcs in jax_cfd/base/equation.py
        """
        alpha = self.params["a"]
        beta = self.params["b"]
        num_steps = len(beta)

        u: List[Optional[GridVariableVector]] = [None] * num_steps
        k: List[Optional[GridVariableVector]] = [None] * num_steps

        # First stage
        u[0] = u0
        k[0] = equation.explicit_terms(u0, dt, t)

        # Intermediate stages
        for i in range(1, num_steps):
            u_star = GridVariableVector(tuple(v.clone() for v in u0))

            for j in range(i):
                if alpha[i - 1][j] != 0:
                    u_star = u_star + dt * alpha[i - 1][j] * k[j]

            u_star = wrap_field_same_bcs(u_star, u0)
            u[i], _ = equation.pressure_projection(u_star)
            k[i] = equation.explicit_terms(u[i], dt, t + i * dt / num_steps)

        u_star = GridVariableVector(tuple(v.clone() for v in u0))
        for j in range(num_steps):
            if beta[j] != 0:
                u_star = u_star + dt * beta[j] * k[j]

        u_star = wrap_field_same_bcs(u_star, u0)
        u_final, p = equation.pressure_projection(u_star)

        u_final = wrap_field_same_bcs(u_final, u0)
        return u_final, p


class PressureProjection(nn.Module):
    def __init__(
        self,
        grid: grids.Grid,
        bc: BoundaryConditions,
        dtype: torch.dtype = torch.float32,
        solver: Union[str, Solver] = "pseudoinverse",
        implementation: Optional[str] = None,
        laplacians: Optional[List[torch.Tensor]] = None,
        **solver_kwargs,
    ):
        """
        Args:
            grid: Grid object describing the spatial domain.
            bc: Boundary conditions for the Laplacian operator (for pressure).
            dtype: Tensor data type. For consistency purpose.
            solver: Solver to use for pressure projection. Can be a string ('cg', 'pseudoinverse', 'multigrid') name or a Solver instance.
            implementation: One of ['fft', 'rfft', 'matmul'].
            circulant: If True, bc is periodical
            laplacians: Precomputed Laplacian operators. If None, they are computed from the grid during initiliazation.
            initial_guess_pressure: Initial guess for pressure. If None, a zero tensor is used.
        """
        super().__init__()
        self.grid = grid
        self.bc = bc
        self.dtype = dtype
        self.implementation = implementation
        solvers._set_laplacian(self, laplacians, grid, bc)
        self.ndim = grid.ndim

        @property
        def inverse(self) -> torch.Tensor:
            return self.solver.inverse

        @property
        def operators(self) -> List[torch.Tensor]:
            """Get the list of 1D Laplacian operators."""
            return [getattr(self.solver, f"laplacian_{i}") for i in range(self.ndim)]

        if isinstance(solver, nn.Module):
            self.solver = solver
        elif isinstance(solver, str):
            if solver in ["conjugate_gradient", "cg"]:
                self.solver = solvers.ConjugateGradient(
                    grid=grid,
                    bc=bc,
                    dtype=dtype,
                    laplacians=laplacians,
                    pure_neumann=True,
                    **solver_kwargs,
                )
            elif solver in ["pseudoinverse", "fft", "rfft", "svd"]:
                self.solver = solvers.PseudoInverse(
                    grid=grid,
                    bc=bc,
                    dtype=dtype,
                    hermitian=True,
                    implementation=implementation,
                    laplacians=laplacians,
                )
            else:
                raise NotImplementedError(f"Unsupported solver: {solver}")

    def forward(self, v: GridVariableVector) -> Tuple[GridVariableVector, GridVariable]:
        """Project velocity to be divergence-free."""
        solver = self.solver.to(v.device)
        if hasattr(self, "q0"):
            # Use the previous pressure as initial guess
            q0 = self.q0.to(v.device)
        else:
            # No previous pressure, use zero as initial guess
            q0 = GridVariable(
                torch.zeros_like(v[0].data, dtype=self.dtype),
                v[0].offset,
                v[0].grid,
                v[0].bc,
            ).to(v.device)
            self.q0 = q0
        _ = grids.consistent_grid(self.grid, *v)
        if self.bc is not None:
            pressure_bc = self.bc
        else:
            pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

        rhs = fdm.divergence(v)
        rhs_transformed = self.rhs_transform(rhs, pressure_bc)
        rhs_inv = solver.solve(rhs_transformed, q0.data)
        q = GridVariable(rhs_inv, rhs.offset, rhs.grid)
        q = pressure_bc.impose_bc(q)
        q_grad = fdm.forward_difference(q) 
        # forward diff has to be used to match the offset of the velocity field
        v_projected = GridVariableVector(
            tuple(u.bc.impose_bc(u - q_g) for u, q_g in zip(v, q_grad))
        )
        self.q0 = q
        # assert v_projected.__len__() == v.__len__()
        return v_projected, q

    @staticmethod
    def rhs_transform(
        f: GridVariable,
        bc: BoundaryConditions,
    ) -> torch.Tensor:
        """Transform the RHS of pressure projection equation for stability."""
        rhs = f.data  # (b, n, m) or (n, m)
        ndim = f.grid.ndim
        for dim in range(ndim):
            if (
                bc.types[dim][0] == boundaries.BCType.NEUMANN
                and bc.types[dim][1] == boundaries.BCType.NEUMANN
            ):
                # Check if we have batched data
                if rhs.ndim > ndim:
                    # For batched data, calculate mean separately for each batch
                    # Keep the batch dimension, reduce over grid dimensions
                    dims = tuple(range(-ndim, 0))
                    mean = torch.mean(rhs, dim=dims, keepdim=True)
                else:
                    # For non-batched data, calculate global mean
                    mean = torch.mean(rhs)
                rhs = rhs - mean
        return rhs


class NavierStokes2DFVMProjection(ProjectionExplicitODE):
    r"""incompressible Navier-Stokes velocity pressure formulation

    Runge-Kutta time stepper for the NSE discretized using a MAC grid FVM with a pressure projection Chorin's method. The x- and y-dofs of the velocity
    are on a staggered grid, which is reflected in the offset attr.

    References:
    - Sanderse, B., & Koren, B. (2012). Accuracy analysis of explicit Runge-Kutta methods applied to the incompressible Navier-Stokes equations. Journal of Computational Physics, 231(8), 3041-3063.
    - Almgren, A. S., Bell, J. B., & Szymczak, W. G. (1996). A numerical method for the incompressible Navier-Stokes equations based on an approximate projection. SIAM Journal on Scientific Computing, 17(2), 358-369.
    - Capuano, F., Coppola, G., Chiatto, M., & de Luca, L. (2016). Approximate projection method for the incompressible Navier-Stokes equations. AIAA journal, 54(7), 2179-2182.

    Args:
        viscosity: 1/Re
        grid: Grid on which the fields are defined
        bcs: Boundary conditions for the velocity field (default: periodic)
        drag: Drag coefficient applied to the velocity field (default: 0.0)
        density: Density of the fluid (default: 1.0)
        convection: Convection term function (default: advection.ConvectionVector)
        pressure_proj: Pressure projection function (default: pressure.PressureProjection)
        forcing: Forcing function applied to the velocity field (default: None)
        step_fn: Runge-Kutta stepper function (default: RKStepper with classic_rk4 method)

    Original implementation in Jax-CFD repository:

    - semi_implicit_navier_stokes in jax_cfd.base.fvm which returns a stepper function `time_stepper(ode, dt)` where `ode` specifies the explicit terms and the pressure projection.
    - The pressure projection is done by calling `pressure.projection` which can solve the solver to solve the Poisson equation \Delta q = div(u).
    - The time_stepper is a wrapper function by jax.named_call(
      navier_stokes_rk()) that implements the various Runge-Kutta method according to the Butcher tableau.
    - navier_stokes_rk() implements Runge-Kutta time-stepping for the NSE using the explicit terms and pressure projection with equation as an input where user needs to specify the explicit terms and pressure projection.

    (Original reference listed in Jax-CFD)
    This class implements the reference method (equations 16-21) from:
    "Fast-Projection Methods for the Incompressible Navier-Stokes Equations"
    Fluids 2020, 5, 222; doi:10.3390/fluids5040222
    """

    def __init__(
        self,
        viscosity: float,
        grid: Grid,
        bcs: Optional[Tuple[boundaries.BoundaryConditions, ...]] = None,
        drag: float = 0.0,
        density: float = 1.0,
        convection: Optional[Callable] = None,
        pressure_proj: Optional[Callable] = None,
        forcing: Optional[ForcingFn] = None,
        step_fn: Optional[RKStepper] = None,
        **kwargs,
    ):
        """
        Args:
            tableau: Tuple (a, b) where a is the coefficient matrix (list of lists of floats)
                    and b is the weight vector (list of floats)
            equation: Navier-Stokes equation to solve
            requires_grad: Whether parameters should be trainable
        """
        super().__init__()
        self.viscosity = viscosity
        self.density = density
        self.grid = grid
        self.bcs = bcs
        self.drag = drag
        self.forcing = forcing
        self.convection = convection
        self.step_fn = step_fn
        self.pressure_proj = pressure_proj
        self._initialize()

    def _initialize(self):
        self.bcs = default(
            self.bcs,
            tuple(
                [
                    boundaries.periodic_boundary_conditions(ndim=self.grid.ndim)
                    for _ in range(self.grid.ndim)
                ]
            ),
        )
        self.pressure_bc = boundaries.get_pressure_bc_from_velocity_bc(bcs=self.bcs)
        self._projection = default(
            self.pressure_proj,
            PressureProjection(
                grid=self.grid,
                bc=self.pressure_bc,
            ),
        )
        self._convect = default(
            self.convection, advection.ConvectionVector(grid=self.grid, bcs=self.bcs)
        )
        self._step_fn = default(self.step_fn, RKStepper.from_method("heun_rk2"))

    def _diffusion(self, v: GridVariableVector) -> GridVariableVector:
        """Returns the diffusion term for the velocity field."""
        alpha = self.viscosity / self.density
        lapv = GridVariableVector(tuple(alpha * fdm.laplacian(u) for u in v))
        return lapv

    def _explicit_terms(
        self, v: GridVariableVector, dt: float, t: Optional[float] = 0.0, **kwargs
    ):
        dv_dt = self._convect(v, v, dt)
        grid = self.grid
        density = self.density
        forcing = self.forcing
        dv_dt += self._diffusion(v)
        if forcing is not None:
            dv_dt += GridVariableVector(forcing(grid, v, t)) / density
        if self.drag > 0.0:
            dv_dt -=  v * self.drag
        return dv_dt

    def explicit_terms(self, *args, **kwargs):
        return self._explicit_terms(*args, **kwargs)

    def pressure_projection(self, *args, **kwargs):
        return self._projection(*args, **kwargs)

    def forward(
        self, u: GridVariableVector, dt: float, t: Optional[float] = 0.0
    ) -> GridVariableVector:
        """Perform one time step.

        Args:
            u: Initial state (velocity field)
            dt: Time step size

        Returns:
            Updated velocity field after one time step
        """

        return self._step_fn(u, dt, self)
