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

"""Tests for torch_cfd.solvers."""

import math

import torch
from absl.testing import absltest, parameterized

from torch_cfd import boundaries, finite_differences as fdm, grids, solvers, test_utils

BCType = grids.BCType


def grid_variable_periodic(data, offset, grid):
    return grids.GridVariable(
        data, offset, grid, bc=boundaries.periodic_boundary_conditions(grid.ndim)
    )


def grid_variable_dirichlet(data, offset, grid):
    return grids.GridVariable(
        data, offset, grid, bc=boundaries.dirichlet_boundary_conditions(grid.ndim)
    )


class SolversTest(test_utils.TestCase):

    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def poisson_random_data(self, bc, offset, shape, batch_size=2, random_state=42):
        """Setup for Poisson equation tests."""
        torch.manual_seed(random_state)
        b = torch.randn((batch_size, *shape), dtype=self.dtype, device=self.device)
        if boundaries.is_bc_pure_neumann_boundary_conditions(bc):
            # For Neumann BC, subtract mean to ensure solvability
            b = b - b.mean(dim=(-2, -1), keepdim=True)
        if boundaries.is_bc_all_periodic_boundary_conditions(bc):
            b = torch.fft.ifftn(b, dim=(-2, -1)).real
        grid = grids.Grid(shape, step=tuple(1.0 / s for s in shape), device=self.device)
        b = grids.GridVariable(b, offset, grid, bc=bc)

        return b, grid

    def poisson_smooth_data(
        self,
        bc,
        offset,
        shape,
        batch_size: int = 2,
        num_modes: int = 10,
        random_state: int = 42,
    ):
        torch.manual_seed(random_state)
        grid = grids.Grid(shape, step=tuple(1.0 / s for s in shape), device=self.device)
        # Create meshgrid
        X, Y = grid.mesh(offset)
        pure_neumann = boundaries.is_bc_pure_neumann_boundary_conditions(bc)
        func = torch.cos if pure_neumann else torch.sin
        # Initialize solution
        u_true = torch.zeros((batch_size, *shape), device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            # Generate random coefficients a_k
            a_k = torch.randn(num_modes, device=self.device, dtype=self.dtype)
            # Sum over modes: u_true = sum(a_k * cos(k*pi*x) * cos(k*pi*y))
            for k in range(1, num_modes + 1):
                b_k = torch.randint(
                    1, num_modes + 1, (1,), device=self.device
                ).item()  # Randomly choose a mode for y
                c_k = torch.randint(
                    1, num_modes + 1, (1,), device=self.device
                ).item()  # Randomly choose a mode for x
                components = (
                    a_k[k - 1] * func(c_k * torch.pi * X) * func(b_k * torch.pi * Y) / k
                )
                u_true[i] += components  # Add batch dimension

        if pure_neumann:
            u_true -= torch.mean(u_true, dim=(-2, -1), keepdim=True)

        u_true = grids.GridVariable(u_true, offset, grid, bc=bc)

        return u_true, grid

    @parameterized.named_parameters(
        dict(
            testcase_name="_2D_periodic_fft",
            shape=(64, 64),
            bc_factory=boundaries.periodic_boundary_conditions,
            solver=solvers.PseudoInverseFFT,
        ),
        dict(
            testcase_name="_2D_dirichlet_matmul",
            shape=(64, 64),
            bc_factory=boundaries.dirichlet_boundary_conditions,
            solver=solvers.PseudoInverseMatmul,
        ),
        dict(
            testcase_name="_2D_periodic_rfft",
            shape=(64, 64),
            bc_factory=boundaries.periodic_boundary_conditions,
            solver=solvers.PseudoInverseRFFT,
        ),
    )
    def test_pseudoinverse_solvers(self, shape, bc_factory, solver):
        """Test that solvers correctly solve Poisson equation."""
        ndim = len(shape)
        h = 1.0 / shape[0]  # Grid spacing
        bc = bc_factory(ndim)
        b, grid = self.poisson_random_data(bc, offset=(0.5,) * ndim, shape=shape)

        # # For periodic BC, subtract mean to ensure solvability
        if boundaries.is_bc_all_periodic_boundary_conditions(bc):
            b.data = b.data - b.data.mean()

        # Create solver
        solver = solver(grid, bc, dtype=self.dtype, tol=1e-12).to(self.device)

        # Solve
        u = solver(b.data, torch.zeros_like(b.data))
        u_var = grids.GridVariable(u, offset=(0.5,) * grid.ndim, grid=grid, bc=bc)

        # Apply Laplacian to solution
        laplacian_u = fdm.laplacian(u_var)

        # Check that Laplacian of solution equals RHS (up to mean for periodic)
        if boundaries.is_bc_all_periodic_boundary_conditions(bc):
            expected = b.data - b.data.mean()
            actual = laplacian_u.data - laplacian_u.data.mean()
        else:
            expected = b.data
            actual = laplacian_u.data

        self.assertAllClose(actual, expected, atol=h**2, rtol=h)

    @parameterized.named_parameters(
        dict(
            testcase_name="_jacobi_2D",
            solver=solvers.Jacobi,
            shape=(32, 32),
            maxiter=2000,
            tol=1e-5,
        ),
        dict(
            testcase_name="_gauss_seidel_2D",
            solver=solvers.GaussSeidel,
            shape=(32, 32),
            maxiter=1000,
            tol=1e-6,
        ),
        dict(
            testcase_name="_cg_2D",
            solver=solvers.ConjugateGradient,
            shape=(32, 32),
            maxiter=100,
            tol=1e-8,
        ),
    )
    def test_iterative_solvers(self, solver, shape, maxiter, tol):
        """Test iterative solvers on Poisson equation."""

        ndim = len(shape)
        bc = boundaries.dirichlet_boundary_conditions(ndim)
        b, grid = self.poisson_random_data(bc, offset=(0.5,) * ndim, shape=shape)

        # Create solver
        solver = solver(grid, bc, dtype=self.dtype, tol=tol, max_iter=maxiter).to(
            self.device
        )

        # Solve
        u_init = torch.zeros_like(b.data)
        u = solver.solve(b.data, u_init)
        u_var = grids.GridVariable(u, offset=(0.5, 0.5), grid=grid, bc=bc)

        # Apply Laplacian to solution
        laplacian_u = fdm.laplacian(u_var)

        # Check that Laplacian of solution equals RHS
        self.assertAllClose(laplacian_u.data, b.data, atol=1e-4, rtol=1e-4)

    @parameterized.named_parameters(
        dict(
            testcase_name="_pseudoinverse_fft",
            solver=solvers.PseudoInverseFFT,
            batch_size=3,
            bc_factory=boundaries.periodic_boundary_conditions,
        ),
        dict(
            testcase_name="_pseudoinverse_rfft",
            solver=solvers.PseudoInverseRFFT,
            batch_size=2,
            bc_factory=boundaries.periodic_boundary_conditions,
        ),
        dict(
            testcase_name="_pseudoinverse_matmul",
            solver=solvers.PseudoInverseMatmul,
            batch_size=4,
            bc_factory=boundaries.dirichlet_boundary_conditions,
        ),
    )
    def test_batch_pseudoinverse(self, solver, batch_size, bc_factory):
        """Test that solvers work with batch dimensions."""
        shape = (16, 16)
        ndim = len(shape)
        bc = bc_factory(ndim)

        torch.manual_seed(9876)
        b, grid = self.poisson_random_data(
            bc, offset=(0.5,) * ndim, shape=shape, batch_size=batch_size
        )

        # Handle periodic BC mean subtraction
        if all(bc_type == BCType.PERIODIC for bc_type in bc.types[0]):
            b = b - torch.mean(b, dim=(-2, -1), keepdim=True)

        solver = solver(grid, bc, dtype=self.dtype, tol=1e-10).to(self.device)

        # Solve batch
        u_init = torch.zeros_like(b.data)
        u_batch = solver.forward(b.data, u_init)

        # Test each item in batch individually
        for i in range(batch_size):
            b_single = b[i]
            u_single = u_batch[i]

            # Create GridVariable and apply Laplacian
            u_single = grids.GridVariable(u_single, offset=(0.5, 0.5), grid=grid, bc=bc)
            laplacian_u = fdm.laplacian(u_single)

            # Check solution
            if all(bc_type == BCType.PERIODIC for bc_type in bc.types[0]):
                expected = b_single.data - torch.mean(b_single.data)
                actual = laplacian_u.data - torch.mean(laplacian_u.data)
            else:
                expected = b_single.data
                actual = laplacian_u.data

            self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)

    def test_laplacian_eigenvalues(self):
        """Test that eigenvalue computation is correct."""
        shape = (8, 8)
        grid = grids.Grid(shape, step=(1.0, 1.0))
        bc = boundaries.periodic_boundary_conditions(grid.ndim)

        solver = solvers.PseudoInverseFFT(grid, bc, dtype=self.dtype).to(self.device)

        # For periodic BC with step=1, eigenvalues should be related to Fourier modes
        expected_eigenvals_1d = []
        for n in shape:
            k = torch.arange(n, dtype=self.dtype, device=self.device) - n // 2
            # Shift to match FFT ordering
            k = torch.fft.fftshift(k)
            # Eigenvalues for central difference on periodic grid
            eigvals = -4 * torch.sin(math.pi * k / n) ** 2
            expected_eigenvals_1d.append(eigvals)

        # Create 2D eigenvalue matrix
        expected_eigenvals_2d = (
            expected_eigenvals_1d[0][:, None] + expected_eigenvals_1d[1][None, :]
        )

        # Get actual eigenvalues from solver (assuming solver stores them)
        inv_eigs = torch.as_tensor(solver.inverse_diag).real
        actual_eigenvals = 1 / inv_eigs

        # Compare eigenvalues (excluding the zero mode for periodic BC)
        mask = torch.abs(expected_eigenvals_2d) > 1e-12
        self.assertAllClose(
            actual_eigenvals[mask], expected_eigenvals_2d[mask], atol=1e-10, rtol=1e-10
        )

        # The solver should handle the zero eigenvalue correctly
        self.assertTrue(hasattr(solver, "inverse_diag"))

    def test_solver_with_zero_rhs(self):
        """Test solver behavior with zero right-hand side."""
        shape = (16, 16)
        grid = grids.Grid(shape, domain=((0, 1), (0, 1)))
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim)

        b = torch.zeros(shape, dtype=self.dtype, device=self.device)

        solver = solvers.PseudoInverseMatmul(grid, bc, dtype=self.dtype).to(self.device)
        u = solver(b, torch.zeros_like(b))

        # Solution should be zero (up to numerical precision)
        self.assertAllClose(u, torch.zeros_like(u), atol=1e-10, rtol=1e-10)

    def test_solver_single_mode(self):
        shape = (64, 64)
        h = 1.0 / shape[0]  # Grid spacing
        grid = grids.Grid(shape, domain=((0, 1), (0, 1)))
        bc = boundaries.periodic_boundary_conditions(grid.ndim)

        X, Y = grid.mesh()
        b = torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
        b = b - b.mean()
        b = b.to(self.device, dtype=self.dtype)

        solver = solvers.PseudoInverseFFT(grid, bc, dtype=self.dtype).to(self.device)
        u = solver(b, torch.zeros_like(b))

        expected_u = b / (-8 * math.pi**2)
        expected_u = expected_u - expected_u.mean()

        self.assertAllClose(u - u.mean(), expected_u, atol=h**2, rtol=h**2)

    @parameterized.named_parameters(
        dict(
            testcase_name="_periodic",
            bc_func=boundaries.periodic_boundary_conditions,
            shape=(16, 16),
            tol=1e-8,
        ),
        dict(
            testcase_name="_dirichlet",
            bc_func=boundaries.dirichlet_boundary_conditions,
            shape=(32, 32),
            tol=1e-10,
        ),
        dict(
            testcase_name="_neumann",
            bc_func=boundaries.neumann_boundary_conditions,
            shape=(32, 32),
            tol=1e-10,
        ),
    )
    def test_pseudoinverse_factory(self, bc_func, shape, tol):
        """Test the PseudoInverse factory class."""
        grid = grids.Grid(shape, domain=((0, 1), (0, 1)))

        # Test automatic selection for periodic BC
        bc = bc_func(grid.ndim)
        is_periodic = all(
            [
                boundaries.is_bc_periodic_boundary_conditions(bc, dim)
                for dim in range(grid.ndim)
            ]
        )
        solver_auto = solvers.PseudoInverse(
            grid,
            bc,
            dtype=self.dtype,
            hermitian=True,
            circulant=is_periodic,
            cutoff=tol,
        )

        if is_periodic:
            self.assertIsInstance(
                solver_auto, (solvers.PseudoInverseFFT, solvers.PseudoInverseRFFT)
            )
        else:
            self.assertIsInstance(solver_auto, solvers.PseudoInverseMatmul)

    @parameterized.named_parameters(
        dict(
            testcase_name="_dirichlet_small",
            bc_func=boundaries.dirichlet_boundary_conditions,
            shape=(64, 64),
            factor=4,
        ),
        dict(
            testcase_name="_dirichlet_medium",
            bc_func=boundaries.dirichlet_boundary_conditions,
            shape=(256, 256),
            factor=8,
        ),
        dict(
            testcase_name="_neumann_small",
            bc_func=boundaries.neumann_boundary_conditions,
            shape=(64, 64),
            factor=4,
        ),
        dict(
            testcase_name="_neumann_medium",
            bc_func=boundaries.neumann_boundary_conditions,
            shape=(256, 256),
            factor=8,
        ),
    )
    def test_conjugate_gradient_convergence(self, bc_func, shape, factor):
        ndim = len(shape)
        h = 1.0 / shape[0]  # Grid spacing
        bc = bc_func(ndim)

        b, grid = self.poisson_random_data(
            bc, offset=(0.5,) * ndim, shape=shape, batch_size=2
        )
        pure_neumann = boundaries.is_bc_pure_neumann_boundary_conditions(bc)
        # Test CG convergence
        tol = h**2 * factor
        rate = (1 - h / factor) / (1 + h / factor)
        max_iter = int(math.ceil(math.log(tol) / math.log(rate)))
        cg = solvers.ConjugateGradient(
            grid,
            bc,
            dtype=self.dtype,
            tol=tol,
            max_iter=max_iter,
            check_iter=1,
            record_residuals=True,
            pure_neumann=pure_neumann,
        ).to(self.device)

        _ = cg.solve(b, torch.zeros_like(b.data))
        residuals = cg.residual_norms.detach().cpu().numpy()
        for i in range(1, len(residuals)):
            if residuals[i] > 0.5:
                self.assertLess(residuals[i], residuals[i - 1] * rate)
        self.assertLess(residuals[-1], tol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_dirichlet_small",
            bc_func=boundaries.dirichlet_boundary_conditions,
            shape=(64, 64),
            level=2,
        ),
        dict(
            testcase_name="_dirichlet_medium",
            bc_func=boundaries.dirichlet_boundary_conditions,
            shape=(128, 128),
            level=3,
        ),
        dict(
            testcase_name="_neumann_small",
            bc_func=boundaries.neumann_boundary_conditions,
            shape=(64, 64),
            level=2,
        ),
        dict(
            testcase_name="_neumann_medium",
            bc_func=boundaries.neumann_boundary_conditions,
            shape=(128, 128),
            level=3,
        ),
    )
    def test_multigrid_convergence(self, bc_func, shape, level):
        """Test multigrid solver if implemented."""
        h = 1.0 / shape[0]  # Grid spacing
        ndim = len(shape)
        bc = bc_func(ndim)
        offset = (0.5,) * ndim
        u_true, grid = self.poisson_smooth_data(
            bc, offset, shape, num_modes=4 * level, batch_size=2
        )
        pure_neumann = boundaries.is_bc_pure_neumann_boundary_conditions(bc)
        tol = h / 8
        factor = (1 - h / 2) / (1 + h / 2)
        solver = solvers.MultigridSolver(
            grid,
            bc,
            dtype=self.dtype,
            tol=tol,
            max_iter=5,
            levels=level,
            record_residuals=True,
            pure_neumann=pure_neumann,
        ).to(self.device)

        b = solver._apply_laplacian(u_true.data)
        if pure_neumann:
            b -= b.mean(dim=(-2, -1), keepdim=True)

        u = solver.solve(b, torch.zeros_like(b))
        residuals = solver.residual_norms.detach().cpu().numpy()
        rel_err = torch.linalg.norm(u_true.data - u) / torch.linalg.norm(u_true.data)
        for i in range(1, len(residuals)):
            if residuals[i] > h:
                self.assertLess(residuals[i], residuals[i - 1] * factor)
        self.assertLess(rel_err.item(), tol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_dirichlet_medium",
            bc_func=boundaries.dirichlet_boundary_conditions,
            shape=(128, 128),
            level=3,
        ),
        dict(
            testcase_name="_dirichlet_large",
            bc_func=boundaries.dirichlet_boundary_conditions,
            shape=(256, 256),
            level=4,
        ),
        dict(
            testcase_name="_neumann_medium",
            bc_func=boundaries.neumann_boundary_conditions,
            shape=(128, 128),
            level=3,
        ),
        dict(
            testcase_name="_neumann_large",
            bc_func=boundaries.neumann_boundary_conditions,
            shape=(256, 256),
            level=4,
        ),
    )
    def test_mg_preconditioned_cg_convergence(self, bc_func, shape, level):
        """Test CG with multigrid preconditioning."""
        h = 1.0 / shape[0]
        ndim = len(shape)
        bc = bc_func(ndim)
        offset = (0.5,) * ndim
        u_true, grid = self.poisson_smooth_data(
            bc, offset, shape, num_modes=4 * level, batch_size=1
        )
        pure_neumann = boundaries.is_bc_pure_neumann_boundary_conditions(bc)
        tol = h
        factor = (1 - h / 8) / (1 + h / 8)

        precond = solvers.MultigridSolver(
            grid,
            bc,
            dtype=self.dtype,
            tol=h,
            max_iter=1,
            pre_smooth=1,
            post_smooth=1,
            levels=level,
        ).to(self.device)

        solver = solvers.ConjugateGradient(
            grid,
            bc,
            dtype=self.dtype,
            tol=tol,
            max_iter=10,
            check_iter=1,
            record_residuals=True,
            preconditioner=precond,
            pure_neumann=pure_neumann,
        ).to(self.device)

        b = solver._apply_laplacian(u_true.data)
        if pure_neumann:
            b -= b.mean(dim=(-2, -1), keepdim=True)

        u = solver.solve(b, torch.zeros_like(b))
        residuals = solver.residual_norms.detach().cpu().numpy()
        rel_err = torch.linalg.norm(u_true.data - u) / torch.linalg.norm(u_true.data)
        for i in range(1, solver.stop_iter):
            if residuals[i] > 8*h:
                self.assertLess(residuals[i], residuals[i - 1] * factor)
        self.assertLess(rel_err.item(), tol)

    @parameterized.named_parameters(
        dict(testcase_name="_128x128", shape=(128, 128)),
        dict(testcase_name="_256x256", shape=(256, 256)),
        dict(testcase_name="_512x512", shape=(512, 512)),
    )
    def test_pseudoinverse_fft(self, shape):
        grid = grids.Grid(shape, domain=((0, 1), (0, 1)))
        bc = boundaries.periodic_boundary_conditions(grid.ndim)

        b = torch.randn(shape, dtype=self.dtype, device=self.device)
        b = b - b.mean()

        solver = solvers.PseudoInverseFFT(grid, bc, dtype=self.dtype).to(self.device)
        u_fft = solver(b, torch.zeros_like(b))

        # Basic correctness check
        u_var = grids.GridVariable(u_fft, offset=(0.5, 0.5), grid=grid, bc=bc)
        laplacian_u = fdm.laplacian(u_var)
        expected = b - b.mean()
        actual = laplacian_u.data - laplacian_u.data.mean()

        self.assertAllClose(actual, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    absltest.main()
