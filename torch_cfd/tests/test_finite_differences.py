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

"""Tests for torch_cfd.finite_difference."""
import math

import torch
from absl.testing import absltest, parameterized
from einops import repeat

from torch_cfd import boundaries, finite_differences as fdm, grids, test_utils

BCType = grids.BCType
Padding = grids.Padding
trim_boundary = fdm.trim_boundary


def grid_variable_periodic(data, offset, grid):
    return grids.GridVariable(
        data, offset, grid, bc=boundaries.periodic_boundary_conditions(grid.ndim)
    )


def grid_variable_dirichlet_constant(data, offset, grid, bc_values=None):
    return grids.GridVariable(
        data,
        offset,
        grid,
        bc=boundaries.dirichlet_boundary_conditions(grid.ndim, bc_values),
    )


def grid_variable_dirichlet_nonhomogeneous(data, offset, grid, bc_values):
    bc = boundaries.DiscreteBoundaryConditions(
        ((BCType.DIRICHLET, BCType.DIRICHLET),) * grid.ndim, bc_values
    )
    return grids.GridVariable(data, offset, grid, bc)


def grid_variable_dirichlet_function_nonhomogeneous(data, offset, grid, bc_funcs):
    bc_types = ((BCType.DIRICHLET, BCType.DIRICHLET),) * grid.ndim
    bc = boundaries.FunctionBoundaryConditions(bc_types, bc_funcs, grid, offset)
    return grids.GridVariable(data, offset, grid, bc)


def grid_variable_dirichlet_nonhomogeneous_and_periodic(
    data, offset, grid, bc_values, periodic_dim=0
):
    bc_dirichlet = (BCType.DIRICHLET, BCType.DIRICHLET)
    bc_periodic = (BCType.PERIODIC, BCType.PERIODIC)
    bc_types = tuple(
        bc_periodic if i == periodic_dim else bc_dirichlet for i in range(grid.ndim)
    )
    bc = boundaries.DiscreteBoundaryConditions(bc_types, bc_values)
    return grids.GridVariable(data, offset, grid, bc)


def grid_variable_vector_batch_from_functions(
    grid, offsets, vfuncs, bc_u, bc_v, batch_size=1
):
    v = []
    for dim, (offset, bc) in enumerate(zip(offsets, (bc_u, bc_v))):
        x, y = grid.mesh(offset)
        data = vfuncs(x, y)
        data = repeat(data[dim], "h w -> b h w", b=batch_size)
        v.append(
            grid_variable_dirichlet_nonhomogeneous_and_periodic(
                data, offset, grid, bc, periodic_dim=dim
            )
        )

    v = grids.GridVariableVector(tuple(v))
    return v


def stack_tensor_matrix(matrix):
    """Stacks a 2D list or tuple of tensors into a rank-4 tensor."""
    return torch.stack([torch.stack(row, dim=0) for row in matrix], dim=0)


class FiniteDifferenceTest(test_utils.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_periodic",
            method=fdm.central_difference,
            shape=(3, 3),
            step=(1.0, 1.0),
            expected_offset=0,
            negative_shift=-1,
            positive_shift=1,
        ),
        dict(
            testcase_name="_backward_difference_periodic",
            method=fdm.backward_difference,
            shape=(2, 3),
            step=(0.1, 0.3),
            expected_offset=-0.5,
            negative_shift=-1,
            positive_shift=0,
        ),
        dict(
            testcase_name="_forward_difference_periodic",
            method=fdm.forward_difference,
            shape=(23, 32),
            step=(0.01, 2.0),
            expected_offset=+0.5,
            negative_shift=0,
            positive_shift=1,
        ),
    )
    def test_finite_difference_indexing(
        self, method, shape, step, expected_offset, negative_shift, positive_shift
    ):
        """Tests finite difference code using explicit indices."""
        grid = grids.Grid(shape, step)
        u = grid_variable_periodic(
            torch.arange(math.prod(shape)).reshape(shape), (0, 0), grid
        )
        actual_x, actual_y = method(u)

        x, y = torch.meshgrid(*[torch.arange(s) for s in shape], indexing="ij")

        diff_x = (
            u.data[torch.roll(x, -positive_shift, dims=0), y]
            - u.data[torch.roll(x, -negative_shift, dims=0), y]
        )
        expected_data_x = diff_x / step[0] / (positive_shift - negative_shift)
        expected_x = grids.GridVariable(expected_data_x, (expected_offset, 0), grid)

        diff_y = (
            u.data[x, torch.roll(y, -positive_shift, dims=1)]
            - u.data[x, torch.roll(y, -negative_shift, dims=1)]
        )
        expected_data_y = diff_y / step[1] / (positive_shift - negative_shift)
        expected_y = grids.GridVariable(expected_data_y, (0, expected_offset), grid)

        self.assertArrayEqual(expected_x, actual_x)
        self.assertArrayEqual(expected_y, actual_y)

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_periodic",
            method=fdm.central_difference,
            shape=(100, 100, 100),
            offset=(0, 0, 0),
            f=lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.sin(z),
            gradf=(
                lambda x, y, z: -torch.sin(x) * torch.cos(y) * torch.sin(z),
                lambda x, y, z: -torch.cos(x) * torch.sin(y) * torch.sin(z),
                lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.cos(z),
            ),
            atol=1e-3,
            rtol=1e-3,
        ),
        dict(
            testcase_name="_backward_difference_periodic",
            method=fdm.backward_difference,
            shape=(100, 100, 100),
            offset=(0, 0, 0),
            f=lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.sin(z),
            gradf=(
                lambda x, y, z: -torch.sin(x) * torch.cos(y) * torch.sin(z),
                lambda x, y, z: -torch.cos(x) * torch.sin(y) * torch.sin(z),
                lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.cos(z),
            ),
            atol=5e-2,
            rtol=1e-3,
        ),
        dict(
            testcase_name="_forward_difference_periodic",
            method=fdm.forward_difference,
            shape=(200, 200, 200),
            offset=(0, 0, 0),
            f=lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.sin(z),
            gradf=(
                lambda x, y, z: -torch.sin(x) * torch.cos(y) * torch.sin(z),
                lambda x, y, z: -torch.cos(x) * torch.sin(y) * torch.sin(z),
                lambda x, y, z: torch.cos(x) * torch.cos(y) * torch.cos(z),
            ),
            atol=5e-2,
            rtol=1e-3,
        ),
    )
    def test_finite_difference_analytic(
        self, method, shape, offset, f, gradf, atol, rtol
    ):
        """Tests finite difference code comparing to the analytice solution."""
        step = tuple([2.0 * torch.pi / s for s in shape])
        grid = grids.Grid(shape, step)
        mesh = grid.mesh()
        u = grid_variable_periodic(f(*mesh), offset, grid)
        expected_grad = torch.stack([df(*mesh) for df in gradf])
        actual_grad = torch.stack([array.data for array in method(u)])
        self.assertAllClose(expected_grad, actual_grad, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_2D_constant",
            shape=(20, 20),
            f=lambda x, y: torch.ones_like(x),
            g=lambda x, y: torch.zeros_like(x),
            atol=1e-3,
            rtol=1e-8,
        ),
        dict(
            testcase_name="_2D_quadratic",
            shape=(21, 21),
            f=lambda x, y: x * (x - 1.0) + y * (y - 1.0),
            g=lambda x, y: 4 * torch.ones_like(x),
            atol=1e-3,
            rtol=1e-8,
        ),
        dict(
            testcase_name="_2D_sine",
            shape=(32, 32),
            f=lambda x, y: torch.sin(math.pi * x) * torch.sin(math.pi * y),
            g=lambda x, y: -2
            * math.pi**2
            * torch.sin(math.pi * x)
            * torch.sin(math.pi * y),
            atol=1 / 32,
            rtol=1e-3,
        ),
    )
    def test_laplacian_periodic(self, shape, f, g, atol, rtol):
        step = tuple([1.0 / s for s in shape])
        grid = grids.Grid(shape, step)
        offset = (0,) * len(shape)
        mesh = grid.mesh(offset)
        u = grid_variable_periodic(f(*mesh), offset, grid)
        expected_laplacian = trim_boundary(grids.GridVariable(g(*mesh), offset, grid))
        actual_laplacian = trim_boundary(fdm.laplacian(u))
        self.assertAllClose(expected_laplacian, actual_laplacian, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_2D_quartic",
            shape=(21, 21),
            f=lambda x, y: x * (x - 1.0) * y * (y - 1.0),
            g=lambda x, y: 2 * y * (y - 1.0) + 2 * x * (x - 1.0),
            atol=1e-2,
            rtol=1e-5,
        ),
        dict(
            testcase_name="_2D_sine",
            shape=(32, 32),
            f=lambda x, y: torch.sin(math.pi * x) * torch.sin(math.pi * y),
            g=lambda x, y: -2
            * math.pi**2
            * torch.sin(math.pi * x)
            * torch.sin(math.pi * y),
            atol=1 / 32,
            rtol=1e-3,
        ),
    )
    def test_laplacian_dirichlet_homogeneous(self, shape, f, g, atol, rtol):
        step = tuple([1.0 / s for s in shape])
        grid = grids.Grid(shape, step)
        offset = (0,) * len(shape)
        mesh = grid.mesh(offset)
        u = grid_variable_dirichlet_constant(f(*mesh), offset, grid)
        expected_laplacian = trim_boundary(grids.GridVariable(g(*mesh), offset, grid))
        actual_laplacian = trim_boundary(fdm.laplacian(u))
        self.assertAllClose(expected_laplacian, actual_laplacian, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_2D_constant",
            shape=(20, 20),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (torch.ones_like(x), torch.ones_like(y)),
            g=lambda x, y: torch.zeros_like(x),
            atol=1e-3,
            rtol=1e-12,
        ),
        dict(
            testcase_name="_2D_quadratic",
            shape=(21, 21),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (x * (x - 1.0), y * (y - 1.0)),
            g=lambda x, y: 2 * x + 2 * y - 2,
            atol=1e-1,
            rtol=1e-3,
        ),
    )
    def test_divergence(self, shape, offsets, f, g, atol, rtol):
        # note: somehow the bcs are incorrectly set but the divergence is still correct
        step = tuple([1.0 / s for s in shape])
        grid = grids.Grid(shape, step)
        v = [
            grid_variable_periodic(f(*grid.mesh(offset))[dim], offset, grid)
            for dim, offset in enumerate(offsets)
        ]
        expected_divergence = trim_boundary(
            grids.GridVariable(g(*grid.mesh()), (0,) * grid.ndim, grid)
        )
        actual_divergence = trim_boundary(fdm.divergence(v))
        self.assertAllClose(
            expected_divergence, actual_divergence, atol=atol, rtol=rtol
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="_solenoidal_8x8",
            shape=(8, 8),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (y, -x),
            g=lambda x, y: -2 * torch.ones_like(x),
            bc_u=((None, None), (torch.zeros(8), torch.ones(8))),
            bc_v=((torch.zeros(8), -torch.ones(8)), (None, None)),
        ),
        dict(
            testcase_name="_solenoidal_32x32",
            shape=(32, 32),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (y, -x),
            g=lambda x, y: -2 * torch.ones_like(x),
            bc_u=((None, None), (torch.zeros(32), torch.ones(32))),
            bc_v=((torch.zeros(32), -torch.ones(32)), (None, None)),
        ),
        dict(
            testcase_name="_wikipedia_example_2d_21x21",
            shape=(21, 21),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (torch.ones_like(x), -(x**2)),
            g=lambda x, y: -2 * x,
            bc_u=((None, None), (torch.ones(21), torch.ones(21))),
            bc_v=((torch.zeros(21), -torch.ones(21)), (None, None)),
        ),
    )
    def test_curl_2d(self, shape, offsets, f, g, bc_u, bc_v):
        step = tuple([1.0 / s for s in shape])
        grid = grids.Grid(shape, step)
        bcvals = [bc_u, bc_v]
        v = [
            grid_variable_dirichlet_nonhomogeneous_and_periodic(
                f(*grid.mesh(offset))[dim], offset, grid, bcval, dim
            )
            for dim, (offset, bcval) in enumerate(zip(offsets, bcvals))
        ]
        result_offset = (0.5, 0.5)
        expected_curl = trim_boundary(
            grids.GridVariable(g(*grid.mesh(result_offset)), result_offset, grid)
        )
        actual_curl = trim_boundary(fdm.curl_2d(v))
        self.assertAllClose(actual_curl, expected_curl, atol=1e-5, rtol=1e-10)

    @parameterized.parameters(
        # Periodic BC
        dict(
            offset=(0,),
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            expected=[[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]],
        ),
        dict(
            offset=(0.5,),
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            expected=[[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]],
        ),
        dict(
            offset=(1.0,),
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            expected=[[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]],
        ),
        # Dirichlet BC
        dict(
            offset=(0,),
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            expected=[[-2, 1, 0], [1, -2, 1], [0, 1, -2]],
        ),
        dict(
            offset=(0.5,),
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            expected=[[-3, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -3]],
        ),
        dict(
            offset=(1.0,),
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            expected=[[-2, 1, 0], [1, -2, 1], [0, 1, -2]],
        ),
        # Neumann BC
        dict(
            offset=(0.5,),
            bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
            expected=[[-1, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -1]],
        ),
        # Neumann-Dirichlet BC
        dict(
            offset=(0.5,),
            bc_types=((BCType.NEUMANN, BCType.DIRICHLET),),
            expected=[[-1, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -3]],
        ),
    )
    def test_laplacian_matrix_w_boundaries(self, offset, bc_types, expected):
        grid = grids.Grid((4,), step=(0.5,))
        bc = boundaries.HomogeneousBoundaryConditions(bc_types)
        actual = fdm.laplacian_matrix_w_boundaries(grid, offset, bc)
        actual = torch.cat([a for a in actual], dim=0)
        expected = 4.0 * torch.tensor(expected)
        self.assertAllClose(actual, expected)


class FiniteDifferenceNonHomogeneousTest(test_utils.TestCase):
    """Test finite difference operations with non-homogeneous boundary conditions."""

    @parameterized.parameters(
        dict(
            shape=(8,),
            offset=(0,),
        ),
        dict(
            shape=(8,),
            offset=(1,),
        ),
        dict(
            shape=(16,),
            offset=(0,),
        ),
        dict(shape=(16,), offset=(1,)),
        dict(shape=(16,), offset=(0.5,)),
    )
    def test_forward_difference_nonhomogeneous_bc_1d(self, shape, offset):
        """Test forward difference operator with non-homogeneous boundary conditions."""
        grid = grids.Grid(shape, domain=((0.0, 1.0),))
        mesh = grid.mesh(offset)

        # Linear function: u = 2x + 1
        # Forward difference should give 2
        # checking the boundary behavior of padding
        u_data = 2 * mesh[0] + 1

        # Non-homogeneous boundary conditions
        bc_values = ((1.0, 3.0),)  # u(0) = 1, u(1) = 3

        u = grids.GridVariable(
            u_data,
            offset,
            grid,
            bc=boundaries.dirichlet_boundary_conditions(grid.ndim, bc_values),
        )
        u = u.impose_bc()

        # the forward diff needs another padding beyond the boundary
        # by default the padding mode is 'extend' or replicate?
        # for the MAC
        # u.shift(+1, 0) gets the replicate padding at the end
        # check the behavior of pad in this case
        forward_diff = trim_boundary(fdm.forward_difference(u, dim=0))

        expected = 2.0 * torch.ones_like(forward_diff.data)

        self.assertAllClose(forward_diff.data, expected, atol=1e-4, rtol=1e-7)

    @parameterized.parameters(
        dict(
            shape=(8,),
            offset=(0,),
        ),
        dict(
            shape=(8,),
            offset=(1,),
        ),
        dict(
            shape=(16,),
            offset=(0,),
        ),
        dict(shape=(16,), offset=(1,)),
        dict(shape=(16,), offset=(0.5,)),
    )
    def test_backward_difference_nonhomogeneous_bc_1d(self, shape, offset):
        """Test backward difference operator with non-homogeneous boundary conditions."""
        grid = grids.Grid(shape, domain=((0.0, 1.0),))
        mesh = grid.mesh(offset)

        u_data = 3 * mesh[0] + 0.5
        bc_values = ((0.5, 3.5),)  # u(0) = 0.5, u(1) = 3.5

        u = grids.GridVariable(
            u_data,
            offset,
            grid,
            bc=boundaries.dirichlet_boundary_conditions(grid.ndim, bc_values),
        )
        u = u.impose_bc()
        backward_diff = trim_boundary(fdm.backward_difference(u, dim=0))

        expected = 3.0 * torch.ones_like(backward_diff.data)

        self.assertAllClose(backward_diff.data, expected, atol=1e-4, rtol=1e-7)

    @parameterized.parameters(
        dict(
            shape=(8,),
            offset=(0,),
        ),
        dict(
            shape=(8,),
            offset=(1,),
        ),
        dict(
            shape=(16,),
            offset=(0,),
        ),
        dict(shape=(16,), offset=(1,)),
        dict(shape=(16,), offset=(0.5,)),
    )
    def test_central_difference_nonhomogeneous_bc_1d(self, shape, offset):
        """Test central difference operator with non-homogeneous boundary conditions."""
        grid = grids.Grid(shape, domain=((0.0, 1.0),))
        mesh = grid.mesh(offset)

        u_data = 4 * mesh[0] + 2

        bc_values = ((2.0, 6.0),)  # u(0) = 2, u(1) = 6

        u = grids.GridVariable(
            u_data,
            offset,
            grid,
            bc=boundaries.dirichlet_boundary_conditions(grid.ndim, bc_values),
        )
        u = u.impose_bc()

        central_diff = trim_boundary(fdm.central_difference(u, dim=0))

        expected = 4.0 * torch.ones_like(central_diff.data)

        self.assertAllClose(central_diff.data, expected, atol=1e-4, rtol=1e-7)

    @parameterized.parameters(
        dict(
            shape=(16, 16),
            offset=(0, 0),
        ),
        dict(
            shape=(16, 16),
            offset=(0, 1),
        ),
        dict(
            shape=(16, 16),
            offset=(1, 0),
        ),
        dict(shape=(16, 16), offset=(1, 1)),
        dict(shape=(32, 32), offset=(0.5, 1)),
        dict(shape=(32, 32), offset=(1, 0.5)),
    )
    def test_central_difference_nonhomogeneous_bc_2d(self, shape, offset):
        """Test central difference operator with non-homogeneous boundary conditions in 2D."""
        grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
        x, y = grid.mesh(offset)
        h = max(grid.step)

        f = lambda x, y: x**2 + 2 * y**2
        fx = lambda x, y: 2 * x
        fy = lambda x, y: 4 * y
        u_data = f(x, y)
        fx_data = fx(x, y)
        fy_data = fy(x, y)

        u = grid_variable_dirichlet_function_nonhomogeneous(u_data, offset, grid, f)
        u = u.impose_bc()

        grad_x = grids.GridVariable(fx_data, offset, grid)
        grad_y = grids.GridVariable(fy_data, offset, grid)

        # Check that gradients are reasonable in interior
        interior_grad_x = trim_boundary(fdm.central_difference(u, dim=0))
        interior_grad_y = trim_boundary(fdm.central_difference(u, dim=1))

        # Get expected gradients at interior points
        expected_grad_x = trim_boundary(grad_x)
        expected_grad_y = trim_boundary(grad_y)

        # Use relaxed tolerance for finite difference approximation
        self.assertAllClose(interior_grad_x, expected_grad_x, atol=6 * h, rtol=h)
        self.assertAllClose(interior_grad_y, expected_grad_y, atol=6 * h, rtol=h)

    @parameterized.named_parameters(
        dict(
            testcase_name="_x_direction_offset_0",
            shape=(8, 4),
            offset=(0, 0),
            f=lambda x, y: x**2,
            g=lambda x, y: 2 * torch.ones_like(x),
            bc_values=((torch.zeros(4), torch.ones(4)), (None, None)),
            periodic_dim=1,
        ),
        dict(
            testcase_name="_x_direction_offset_1",
            shape=(8, 8),
            offset=(1, 0),
            f=lambda x, y: x**2,
            g=lambda x, y: 2 * torch.ones_like(x),
            bc_values=((torch.zeros(8), torch.ones(8)), (None, None)),
            periodic_dim=1,
        ),
        dict(
            testcase_name="_y_direction_offset_0",
            shape=(8, 4),
            offset=(0, 0),
            f=lambda x, y: y**2,
            g=lambda x, y: 2 * torch.ones_like(y),
            bc_values=((None, None), (torch.zeros(8), torch.ones(8))),
            periodic_dim=0,
        ),
        dict(
            testcase_name="_y_direction_offset_1",
            shape=(4, 4),
            offset=(0, 1),
            f=lambda x, y: y**2,
            g=lambda x, y: 2 * torch.ones_like(y),
            bc_values=((None, None), (torch.zeros(4), torch.ones(4))),
            periodic_dim=0,
        ),
    )
    def test_laplacian_dirichlet_nonhomogeneous(
        self, shape, offset, f, g, bc_values, periodic_dim
    ):
        grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
        mesh = grid.mesh(offset)

        # u = x^2, Laplacian of u is 2
        u_data = f(*mesh)
        expected_laplacian = trim_boundary(grids.GridVariable(g(*mesh), offset, grid))

        # Create GridVariable with non-homogeneous Dirichlet BCs
        u = grid_variable_dirichlet_nonhomogeneous_and_periodic(
            u_data, offset, grid, bc_values, periodic_dim=periodic_dim
        )
        # u = u.bc.impose_bc(u, mode=Padding.EXTEND)
        u = u.impose_bc()

        # Compute Laplacian using finite differences
        actual_laplacian = trim_boundary(fdm.laplacian(u))

        # Use relaxed tolerance due to boundary effects
        self.assertAllClose(actual_laplacian, expected_laplacian, atol=1e-2, rtol=1e-2)

    @parameterized.named_parameters(
        dict(
            testcase_name="_constant_offset_0_0",
            shape=(4, 4),
            offset=(0, 0),
            f=lambda x, y: torch.ones_like(x),
            g=lambda x, y: torch.zeros_like(x),
            bc_values=((torch.ones(4), torch.ones(4)), (torch.ones(4), torch.ones(4))),
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_constant_offset_0_1",
            shape=(4, 4),
            offset=(0, 1),
            f=lambda x, y: torch.ones_like(x),
            g=lambda x, y: torch.zeros_like(x),
            bc_values=((torch.ones(4), torch.ones(4)), (torch.ones(4), torch.ones(4))),
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_constant_offset_1_0",
            shape=(4, 4),
            offset=(1, 0),
            f=lambda x, y: torch.ones_like(x),
            g=lambda x, y: torch.zeros_like(x),
            bc_values=((torch.ones(4), torch.ones(4)), (torch.ones(4), torch.ones(4))),
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_constant_offset_1_1",
            shape=(4, 4),
            offset=(1, 1),
            f=lambda x, y: torch.ones_like(x),
            g=lambda x, y: torch.zeros_like(x),
            bc_values=((torch.ones(4), torch.ones(4)), (torch.ones(4), torch.ones(4))),
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_linear_offset_0_0",
            shape=(8, 8),
            offset=(0, 0),
            f=lambda x, y: x + 2 * y,
            g=lambda x, y: torch.zeros_like(x),
            bc_values=(
                (torch.linspace(0, 2, 9)[:-1], torch.linspace(1, 3, 9)[:-1]),
                (torch.linspace(0, 1, 9)[:-1], torch.linspace(2, 3, 9)[:-1]),
            ),  # ((2y, 1+2y), (x, 2+x))
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_linear_offset_0_1",
            shape=(8, 8),
            offset=(0, 1),
            f=lambda x, y: x + 2 * y,
            g=lambda x, y: torch.zeros_like(x),
            bc_values=(
                (torch.linspace(0, 2, 9)[1:], torch.linspace(1, 3, 9)[1:]),
                (torch.linspace(0, 1, 9)[:-1], torch.linspace(2, 3, 9)[:-1]),
            ),  # ((2y, 1+2y), (x, 2+x))
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_linear_offset_1_0",
            shape=(8, 8),
            offset=(1, 0),
            f=lambda x, y: x + 2 * y,
            g=lambda x, y: torch.zeros_like(x),
            bc_values=(
                (torch.linspace(0, 2, 9)[:-1], torch.linspace(1, 3, 9)[:-1]),
                (torch.linspace(0, 1, 9)[1:], torch.linspace(2, 3, 9)[1:]),
            ),  # ((2y, 1+2y), (x, 2+x))
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_linear_offset_1_1",
            shape=(8, 8),
            offset=(1, 1),
            f=lambda x, y: x + 2 * y,
            g=lambda x, y: torch.zeros_like(x),
            bc_values=(
                (torch.linspace(0, 2, 9)[1:], torch.linspace(1, 3, 9)[1:]),
                (torch.linspace(0, 1, 9)[1:], torch.linspace(2, 3, 9)[1:]),
            ),  # ((2y, 1+2y), (x, 2+x))
            atol=1e-3,
            rtol=1e-10,
        ),
    )
    def test_laplacian_dirichlet_nonhomogeneous_2d(
        self, shape, offset, f, g, bc_values, atol, rtol
    ):
        grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
        mesh = grid.mesh(offset)

        # Create GridVariable with non-homogeneous Dirichlet BCs
        u_data = f(*mesh)
        expected_laplacian = trim_boundary(grids.GridVariable(g(*mesh), offset, grid))

        u = grid_variable_dirichlet_nonhomogeneous(u_data, offset, grid, bc_values)
        u = u.impose_bc()

        # Compute Laplacian using finite differences
        actual_laplacian = trim_boundary(fdm.laplacian(u))

        # Use relaxed tolerance due to boundary effects
        self.assertAllClose(actual_laplacian, expected_laplacian, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_linear_cell_center",
            shape=(16, 16),
            offset=(0.5, 0.5),
            f=lambda x, y: x + 2 * y,
            g=lambda x, y: torch.zeros_like(x),
            atol=1e-6,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_linear_vertical_edge_center",
            shape=(16, 16),
            offset=(1.0, 0.5),
            f=lambda x, y: 2 * x + y,
            g=lambda x, y: torch.zeros_like(x),
            atol=1e-6,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_linear_horizontal_edge_center",
            shape=(16, 16),
            offset=(0.5, 1.0),
            f=lambda x, y: x + 2 * y,
            g=lambda x, y: torch.zeros_like(x),
            atol=1e-6,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_quadratic_lower_left_corner",
            shape=(32, 32),
            offset=(0, 0),
            f=lambda x, y: x**2 + 2 * y**2 + x * y,
            g=lambda x, y: (2 + 4) * torch.ones_like(x),
            atol=1 / 32,
            rtol=1e-2,
        ),
        dict(
            testcase_name="_quadratic_vertical_edge_center",
            shape=(32, 32),
            offset=(1, 0.5),
            f=lambda x, y: 3 * x**2 + y**2 - x * y,
            g=lambda x, y: (6 + 2) * torch.ones_like(x),
            atol=1 / 32,
            rtol=1e-2,
        ),
        dict(
            testcase_name="_quadratic_lower_right_corner",
            shape=(32, 32),
            offset=(1.0, 0),
            f=lambda x, y: 0.5 * x**2 + 1.5 * y**2 + 2 * x * y,
            g=lambda x, y: (1 + 3) * torch.ones_like(x),
            atol=1 / 32,
            rtol=1e-2,
        ),
        dict(
            testcase_name="_quadratic_upper_right_corner",
            shape=(32, 32),
            offset=(1, 1),
            f=lambda x, y: 2 * x**2 + 0.5 * y**2 - 0.5 * x * y + x + y,
            g=lambda x, y: (4 + 1) * torch.ones_like(x),
            atol=1 / 32,
            rtol=1e-2,
        ),
        dict(
            testcase_name="_quadratic_horizontal_edge_center",
            shape=(32, 32),
            offset=(0.5, 1.0),
            f=lambda x, y: x**2 + y**2 + 3 * x * y + 2 * x - y,
            g=lambda x, y: (2 + 2) * torch.ones_like(x),
            atol=1 / 32,
            rtol=1e-2,
        ),
        dict(
            testcase_name="_quadratic_cell_center",
            shape=(16, 16),
            offset=(0.5, 0.5),
            f=lambda x, y: 4 * x**2 + 3 * y**2 + 2 * x * y,
            g=lambda x, y: (8.0 + 6.0) * torch.ones_like(x),
            atol=1 / 16,
            rtol=1e-2,
        ),
    )
    def test_laplacian_dirichlet_function_nonhomogeneous_2d(
        self, shape, offset, f, g, atol, rtol
    ):
        """Test Laplacian with FunctionBoundaryConditions using quadratic functions."""
        grid = grids.Grid(shape, domain=((-1.0, 1.0), (-1.0, 1.0)))
        mesh = grid.mesh(offset)

        # Create GridVariable with function-based non-homogeneous Dirichlet BCs
        u_data = f(*mesh)
        expected_laplacian = trim_boundary(grids.GridVariable(g(*mesh), offset, grid))

        # Use FunctionBoundaryConditions instead of discrete values
        u = grid_variable_dirichlet_function_nonhomogeneous(u_data, offset, grid, f)
        u = u.impose_bc()

        # Compute Laplacian using finite differences
        actual_laplacian = trim_boundary(fdm.laplacian(u))

        self.assertAllClose(actual_laplacian, expected_laplacian, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        dict(
            testcase_name="_laplacian_consistency_8x8",
            shape=(8, 8),
        ),
        dict(
            testcase_name="_laplacian_consistency_8x16",
            shape=(8, 16),
        ),
        dict(
            testcase_name="_laplacian_consistency_16x8",
            shape=(16, 8),
        ),
        dict(
            testcase_name="_laplacian_consistency_32x32",
            shape=(32, 32),
        ),
    )
    def test_laplacian_consistency(self, shape):
        """Test that Laplacian computation is consistent across different grid resolutions."""
        f = lambda x, y: 0.25 * (x**2 + y**2)
        offsets = [(0, 0), (0.5, 1), (1.0, 0.5), (0.5, 0.5), (1.0, 1.0)]

        for offset in offsets:
            grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
            u_data = f(*grid.mesh(offset))
            u = grid_variable_dirichlet_function_nonhomogeneous(u_data, offset, grid, f)
            u = u.impose_bc()

            laplacian_result = fdm.laplacian(u)

            # Check interior points only where we expect Laplacian ≈ 4
            interior_laplacian = trim_boundary(laplacian_result).data
            expected_interior = torch.ones_like(interior_laplacian)

            self.assertAllClose(
                interior_laplacian, expected_interior, atol=1e-3, rtol=1e-2
            )


class FiniteDifferenceBatchTest(test_utils.TestCase):
    """Test finite difference operations with batch dimensions in 2D."""

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_batch",
            method=fdm.central_difference,
            batch_size=4,
            shape=(32, 32),
            step=(0.1, 0.1),
            expected_offset=0,
        ),
        dict(
            testcase_name="_backward_difference_batch",
            method=fdm.backward_difference,
            batch_size=2,
            shape=(16, 24),
            step=(0.2, 0.15),
            expected_offset=-0.5,
        ),
        dict(
            testcase_name="_forward_difference_batch",
            method=fdm.forward_difference,
            batch_size=3,
            shape=(20, 20),
            step=(0.05, 0.05),
            expected_offset=+0.5,
        ),
    )
    def test_finite_difference_batch_preserves_shape(
        self, method, batch_size, shape, step, expected_offset
    ):
        """Test that finite difference operations preserve batch dimensions."""
        grid = grids.Grid(shape, step)

        # Create batched data: (batch_size, *shape)
        batched_data = torch.randn(batch_size, *shape)
        u = grid_variable_periodic(batched_data, (0, 0), grid)

        # Apply finite difference
        grad_x, grad_y = method(u)

        # Check that batch dimension is preserved
        self.assertEqual(grad_x.data.shape, (batch_size, *shape))
        self.assertEqual(grad_y.data.shape, (batch_size, *shape))

        # Check offsets are correct
        self.assertEqual(grad_x.offset, (expected_offset, 0))
        self.assertEqual(grad_y.offset, (0, expected_offset))

    @parameterized.named_parameters(
        dict(
            testcase_name="_central_difference_coarse",
            method=fdm.central_difference,
            batch_size=2,
            shape=(40, 60),
            f=lambda x, y: torch.sin(x) * torch.cos(y),
            gradf=(
                lambda x, y: torch.cos(x) * torch.cos(y),
                lambda x, y: -torch.sin(x) * torch.sin(y),
            ),
            atol=2 * math.pi / 40,
            rtol=1 / 40,
        ),
        dict(
            testcase_name="_forward_difference",
            method=fdm.forward_difference,
            batch_size=3,
            shape=(128, 256),
            f=lambda x, y: torch.sin(x) * torch.cos(y),
            gradf=(
                lambda x, y: torch.cos(x) * torch.cos(y),
                lambda x, y: -torch.sin(x) * torch.sin(y),
            ),
            atol=2 * math.pi / 128,
            rtol=1 / 128,
        ),
        dict(
            testcase_name="_central_difference_coarse_fine",
            method=fdm.central_difference,
            batch_size=8,
            shape=(1024, 2048),
            f=lambda x, y: torch.sin(x) * torch.cos(y),
            gradf=(
                lambda x, y: torch.cos(x) * torch.cos(y),
                lambda x, y: -torch.sin(x) * torch.sin(y),
            ),
            atol=2 * math.pi / 1024,
            rtol=1 / 1024,
        ),
    )
    def test_finite_difference_batch_analytic(
        self, method, batch_size, shape, f, gradf, atol, rtol
    ):
        """Test finite difference on batched data against analytical solutions."""
        step = tuple([2.0 * math.pi / s for s in shape])
        grid = grids.Grid(shape, step)
        mesh = grid.mesh()

        # Create batched data by repeating the same function
        # In practice, each batch element could be different
        single_data = f(*mesh)
        batched_data = repeat(single_data, "h w -> b h w", b=batch_size)

        u = grid_variable_periodic(batched_data, (0, 0), grid)

        # Compute gradients
        actual_grad_x, actual_grad_y = method(u)

        # Expected gradients (also batched)
        expected_grad_x = repeat(gradf[0](*mesh), "h w -> b h w", b=batch_size)
        expected_grad_y = repeat(gradf[1](*mesh), "h w -> b h w", b=batch_size)

        self.assertAllClose(expected_grad_x, actual_grad_x.data, atol=atol, rtol=rtol)
        self.assertAllClose(expected_grad_y, actual_grad_y.data, atol=atol, rtol=rtol)

    def test_laplacian_batch(self):
        """Test Laplacian operator with batch dimensions."""
        batch_size = 2
        shape = (32, 64)
        grid = grids.Grid(shape, domain=((-1.0, 1.0), (-1.0, 1.0)))
        offset = (0, 0)

        f = lambda x, y: x**2 + y**2
        mesh = grid.mesh(offset)
        single_data = f(*mesh)
        batched_data = repeat(single_data, "h w -> b h w", b=batch_size)

        u = grid_variable_dirichlet_function_nonhomogeneous(batched_data, offset, grid, f)
        actual_laplacian = fdm.laplacian(u)

        # Expected Laplacian is 4 everywhere
        expected_laplacian = 4 * torch.ones(batch_size, *shape)

        # Trim boundary for comparison
        trimmed_actual = trim_boundary(actual_laplacian)
        trimmed_expected = trim_boundary(
            grids.GridVariable(expected_laplacian, offset, grid)
        )

        self.assertAllClose(
            trimmed_expected.data, trimmed_actual.data, atol=1e-3, rtol=1e-8
        )

    def test_laplacian_batch_analytic(self):
        """Test Laplacian operator on batched data against Laplacian on a single data."""
        batch_size = 3
        shape = (128, 128)
        step = (2 * math.pi / 128, 2 * math.pi / 128)
        grid = grids.Grid(shape, step)

        mesh = grid.mesh()
        single_data = torch.sin(mesh[0]) * torch.cos(mesh[1])
        batched_data = repeat(single_data, "h w -> b h w", b=batch_size)

        u_single = grid_variable_periodic(single_data, (0, 0), grid)
        u_batch = grid_variable_periodic(batched_data, (0, 0), grid)
        single_laplacian = fdm.laplacian(u_single)
        batch_laplacian = fdm.laplacian(u_batch)

        # Trim boundary for comparison
        trimmed_single = trim_boundary(single_laplacian)
        trimmed_batch = trim_boundary(batch_laplacian)

        for i in range(batch_size):
            self.assertAllClose(
                trimmed_single.data, trimmed_batch.data[i], atol=1e-8, rtol=1e-12
            )

    def test_divergence_batch(self):
        """Test divergence operator with batch dimensions."""
        batch_size = 3
        shape = (16, 16)
        step = (0.1, 0.1)
        grid = grids.Grid(shape, step)
        offsets = ((0.5, 0), (0, 0.5))

        # Test vector field: v = (x, y), so divergence should be 2
        mesh_x = grid.mesh(offsets[0])
        mesh_y = grid.mesh(offsets[1])

        # Create batched vector components
        vx_single = mesh_x[0]  # x component
        vy_single = mesh_y[1]  # y component

        vx_batched = repeat(vx_single, "h w -> b h w", b=batch_size)
        vy_batched = repeat(vy_single, "h w -> b h w", b=batch_size)

        v = [
            grid_variable_periodic(vx_batched, offsets[0], grid),
            grid_variable_periodic(vy_batched, offsets[1], grid),
        ]

        actual_divergence = fdm.divergence(v)

        # Expected divergence is 2 everywhere
        expected_divergence = 2 * torch.ones(batch_size, *shape)

        # Trim boundary for comparison
        trimmed_actual = trim_boundary(actual_divergence)
        trimmed_expected = trim_boundary(
            grids.GridVariable(expected_divergence, (0, 0), grid)
        )

        self.assertAllClose(
            trimmed_expected.data, trimmed_actual.data, atol=1e-2, rtol=1e-8
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="_solenoidal_8x8",
            shape=(8, 8),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (y, -x),
            g=lambda x, y: -2 * torch.ones_like(x),
            bc_u=((None, None), (torch.zeros(8), torch.ones(8))),
            bc_v=((torch.zeros(8), -torch.ones(8)), (None, None)),
        ),
        dict(
            testcase_name="_solenoidal_32x32",
            shape=(32, 32),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (y, -x),
            g=lambda x, y: -2 * torch.ones_like(x),
            bc_u=((None, None), (torch.zeros(32), torch.ones(32))),
            bc_v=((torch.zeros(32), -torch.ones(32)), (None, None)),
        ),
        dict(
            testcase_name="_wikipedia_example_2d_21x21",
            shape=(21, 21),
            offsets=((0.5, 0), (0, 0.5)),
            f=lambda x, y: (torch.ones_like(x), -(x**2)),
            g=lambda x, y: -2 * x,
            bc_u=((None, None), (torch.ones(21), torch.ones(21))),
            bc_v=((torch.zeros(21), -torch.ones(21)), (None, None)),
        ),
    )
    def test_curl_2d_batch(self, shape, offsets, f, g, bc_u, bc_v):
        """Test 2D curl operator with batch dimensions."""
        batch_size = 2
        grid = grids.Grid(shape, domain=((0, 1), (0, 1)))
        offsets = ((0.5, 0), (0, 0.5))

        v = grid_variable_vector_batch_from_functions(
            grid, offsets, f, bc_u, bc_v, batch_size=batch_size
        )

        actual_curl = trim_boundary(fdm.curl_2d(v))

        # Expected curl is 2 everywhere
        result_offset = (0.5, 0.5)
        expected_curl = g(*grid.mesh(result_offset))
        expected_curl = repeat(expected_curl, "h w -> b h w", b=batch_size)
        expected_curl = trim_boundary(
            grids.GridVariable(expected_curl, result_offset, grid)
        )

        self.assertAllClose(actual_curl.data, expected_curl.data, atol=1e-2, rtol=1e-8)

    def test_batch_consistency_across_operations(self):
        """Test that batch operations are consistent across different batch sizes."""
        shape = (24, 24)
        step = (0.05, 0.05)
        grid = grids.Grid(shape, step)

        # Create test function
        mesh = grid.mesh()
        single_data = torch.sin(mesh[0]) * torch.cos(mesh[1])

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            batched_data = repeat(single_data, "h w -> b h w", b=batch_size)
            u = grid_variable_periodic(batched_data, (0, 0), grid)

            # Apply central difference
            grad_x = fdm.central_difference(u, dim=0)
            grad_y = fdm.central_difference(u, dim=1)

            # Each batch element should be identical
            for i in range(1, batch_size):
                self.assertAllClose(grad_x.data[0], grad_x.data[i])
                self.assertAllClose(grad_y.data[0], grad_y.data[i])

    @parameterized.named_parameters(
        dict(
            testcase_name="_quadratic_1_batch",
            batch_size=3,
            shape=(16, 16),
            offset=(1, 0),
            f=lambda x, y: x**2 + 2 * y**2 + x * y,
            g=lambda x, y: (2 + 4) * torch.ones_like(x),
            atol=1 / 16,
            rtol=1e-2,
        ),
        dict(
            testcase_name="_quadratic_2_batch",
            batch_size=4,
            shape=(32, 32),
            offset=(0, 1),
            f=lambda x, y: 3 * x**2 + y**2 - x * y,
            g=lambda x, y: (6 + 2) * torch.ones_like(x),
            atol=1 / 32,
            rtol=1e-2,
        ),
        dict(
            testcase_name="_quadratic_and_trig_batch",
            batch_size=2,
            shape=(32, 32),
            offset=(1, 0),
            f=lambda x, y: 4 * x**2
            + 3 * y**2
            + 2 * x * y
            + torch.sin(torch.pi * x) * torch.sin(torch.pi * y)/2,
            g=lambda x, y: 14 * torch.ones_like(x)
            - torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y),
            atol=1 / 32,
            rtol=1e-2,
        ),
    )
    def test_laplacian_dirichlet_function_nonhomogeneous_batch(
        self, batch_size, shape, offset, f, g, atol, rtol
    ):
        """Test Laplacian with FunctionBoundaryConditions using quadratic functions with batch dimensions."""
        grid = grids.Grid(shape, domain=((-1.0, 1.0), (-1.0, 1.0)))
        mesh = grid.mesh(offset)

        # Create batched data by repeating the same function
        single_u_data = f(*mesh)
        batched_u_data = repeat(single_u_data, "h w -> b h w", b=batch_size)

        single_expected = g(*mesh)
        batched_expected = repeat(single_expected, "h w -> b h w", b=batch_size)
        expected_laplacian = trim_boundary(
            grids.GridVariable(batched_expected, offset, grid)
        )

        # Use FunctionBoundaryConditions with batched data
        u = grid_variable_dirichlet_function_nonhomogeneous(
            batched_u_data, offset, grid, f
        )
        u = u.impose_bc()

        # Compute Laplacian using finite differences
        actual_laplacian = trim_boundary(fdm.laplacian(u))

        # Check that batch dimension is preserved
        self.assertEqual(actual_laplacian.data.shape[0], batch_size)

        # Check accuracy for each batch element
        self.assertAllClose(
            actual_laplacian.data, expected_laplacian.data, atol=atol, rtol=rtol
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="_linear_discrete_bc_batch",
            batch_size=2,
            shape=(8, 8),
            offset=(0, 0),
            f=lambda x, y: x + 2 * y,
            g=lambda x, y: torch.zeros_like(x),
            bc_values_func=lambda: (
                (torch.linspace(0, 2, 9)[:-1], torch.linspace(1, 3, 9)[:-1]),
                (torch.linspace(0, 1, 9)[:-1], torch.linspace(2, 3, 9)[:-1]),
            ),
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_constant_discrete_bc_batch",
            batch_size=4,
            shape=(12, 12),
            offset=(1, 1),
            f=lambda x, y: torch.ones_like(x),
            g=lambda x, y: torch.zeros_like(x),
            bc_values_func=lambda: (
                (torch.ones(12), torch.ones(12)),
                (torch.ones(12), torch.ones(12)),
            ),
            atol=1e-3,
            rtol=1e-10,
        ),
        dict(
            testcase_name="_quadratic_discrete_bc_batch",
            batch_size=3,
            shape=(16, 16),
            offset=(0.5, 0.5),
            f=lambda x, y: x**2 + y**2,
            g=lambda x, y: 4 * torch.ones_like(x),
            bc_values_func=lambda: (
                (torch.linspace(0, 2, 17)[1:-1], torch.linspace(1, 3, 17)[1:-1]),
                (torch.linspace(0, 1, 17)[1:-1], torch.linspace(1, 2, 17)[1:-1]),
            ),
            atol=1e-2,
            rtol=1e-2,
        ),
    )
    def test_laplacian_dirichlet_discrete_nonhomogeneous_batch(
        self, batch_size, shape, offset, f, g, bc_values_func, atol, rtol
    ):
        """Test Laplacian with discrete non-homogeneous Dirichlet BCs with batch dimensions."""
        grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
        mesh = grid.mesh(offset)

        # Create batched data
        single_u_data = f(*mesh)
        batched_u_data = repeat(single_u_data, "h w -> b h w", b=batch_size)

        single_expected = g(*mesh)
        batched_expected = repeat(single_expected, "h w -> b h w", b=batch_size)
        expected_laplacian = trim_boundary(
            grids.GridVariable(batched_expected, offset, grid)
        )

        # Get boundary condition values
        bc_values = bc_values_func()

        # Create GridVariable with batched non-homogeneous Dirichlet BCs
        u = grid_variable_dirichlet_nonhomogeneous(
            batched_u_data, offset, grid, bc_values
        )
        u = u.impose_bc()

        # Compute Laplacian using finite differences
        actual_laplacian = trim_boundary(fdm.laplacian(u))

        # Check that batch dimension is preserved
        self.assertEqual(actual_laplacian.data.shape[0], batch_size)

        # Check accuracy
        self.assertAllClose(
            actual_laplacian.data, expected_laplacian.data, atol=atol, rtol=rtol
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="_quadratic_gradient_batch",
            batch_size=2,
            shape=(16, 16),
            offset=(0, 0),
            f=lambda x, y: x**2 + 2 * y**2,
            fx=lambda x, y: 2 * x,
            fy=lambda x, y: 4 * y,
        ),
        dict(
            testcase_name="_mixed_quadratic_gradient_batch",
            batch_size=3,
            shape=(32, 32),
            offset=(0.5, 1),
            f=lambda x, y: 3 * x**2 + y**2 + x * y,
            fx=lambda x, y: 6 * x + y,
            fy=lambda x, y: 2 * y + x,
        ),
        dict(
            testcase_name="_cubic_gradient_batch",
            batch_size=4,
            shape=(24, 24),
            offset=(1, 0.5),
            f=lambda x, y: x**3 + y**3 + x * y**2,
            fx=lambda x, y: 3 * x**2 + y**2,
            fy=lambda x, y: 3 * y**2 + 2 * x * y,
        ),
    )
    def test_central_difference_function_nonhomogeneous_batch(
        self, batch_size, shape, offset, f, fx, fy
    ):
        """Test central difference with FunctionBoundaryConditions with batch dimensions."""
        grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
        x, y = grid.mesh(offset)
        h = max(grid.step)

        # Create batched data
        single_u_data = f(x, y)
        batched_u_data = repeat(single_u_data, "h w -> b h w", b=batch_size)

        single_fx_data = fx(x, y)
        single_fy_data = fy(x, y)
        batched_fx_data = repeat(single_fx_data, "h w -> b h w", b=batch_size)
        batched_fy_data = repeat(single_fy_data, "h w -> b h w", b=batch_size)

        u = grid_variable_dirichlet_function_nonhomogeneous(
            batched_u_data, offset, grid, f
        )
        u = u.impose_bc()

        expected_grad_x = grids.GridVariable(batched_fx_data, offset, grid)
        expected_grad_y = grids.GridVariable(batched_fy_data, offset, grid)

        # Check that gradients are reasonable in interior
        interior_grad_x = trim_boundary(fdm.central_difference(u, dim=0))
        interior_grad_y = trim_boundary(fdm.central_difference(u, dim=1))

        # Get expected gradients at interior points
        expected_grad_x_interior = trim_boundary(expected_grad_x)
        expected_grad_y_interior = trim_boundary(expected_grad_y)

        # Check that batch dimension is preserved
        self.assertEqual(interior_grad_x.data.shape[0], batch_size)
        self.assertEqual(interior_grad_y.data.shape[0], batch_size)

        # Use relaxed tolerance for finite difference approximation
        self.assertAllClose(
            interior_grad_x.data, expected_grad_x_interior.data, atol=6 * h, rtol=h
        )
        self.assertAllClose(
            interior_grad_y.data, expected_grad_y_interior.data, atol=6 * h, rtol=h
        )

        # Test batch consistency: each batch element should be identical
        for i in range(1, batch_size):
            self.assertAllClose(
                interior_grad_x.data[0], interior_grad_x.data[i], atol=1e-12, rtol=1e-15
            )
            self.assertAllClose(
                interior_grad_y.data[0], interior_grad_y.data[i], atol=1e-12, rtol=1e-15
            )


if __name__ == "__main__":
    absltest.main()
