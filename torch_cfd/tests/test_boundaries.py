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
"""Tests for torch_cfd.boundaries."""

from functools import partial

import math
import numpy as np
import torch
from absl.testing import absltest, parameterized

from torch_cfd import boundaries, grids, test_utils

BCType = boundaries.BCType
Padding = boundaries.Padding

tensor = partial(torch.tensor, dtype=torch.float32)


class ConstantBoundaryConditionsTest(test_utils.TestCase):

    def test_bc_utilities(self):
        with self.subTest("periodic"):
            bc = boundaries.periodic_boundary_conditions(ndim=2)
            self.assertEqual(
                bc.types, (("periodic", "periodic"), ("periodic", "periodic"))
            )

        with self.subTest("dirichlet"):
            bc = boundaries.dirichlet_boundary_conditions(ndim=2)
            self.assertEqual(
                bc.types, (("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"))
            )

        with self.subTest("neumann"):
            bc = boundaries.neumann_boundary_conditions(ndim=2)
            self.assertEqual(bc.types, (("neumann", "neumann"), ("neumann", "neumann")))

        with self.subTest("periodic_and_neumann"):
            bc = boundaries.periodic_and_neumann_boundary_conditions()
            self.assertEqual(
                bc.types, (("periodic", "periodic"), ("neumann", "neumann"))
            )

    def test_periodic_shift(self):
        shape = (4,)
        offset = (0,)
        step = (1.0,)
        data = tensor([11, 12, 13, 14])
        grid = grids.Grid(shape, step)
        bc = boundaries.periodic_boundary_conditions(grid.ndim)
        array = grids.GridVariable(data, offset, grid, bc)

        for shift, expected in [
            (-2, tensor([13, 14, 11, 12])),
            (-1, tensor([14, 11, 12, 13])),
            (0, tensor([11, 12, 13, 14])),
            (1, tensor([12, 13, 14, 11])),
            (2, tensor([13, 14, 11, 12])),
        ]:
            with self.subTest(shift=shift):
                shifted = bc.shift(array, shift, dim=0)
                expected_offset = (offset[0] + shift,)
                self.assertArrayEqual(
                    shifted, grids.GridVariable(expected, expected_offset, grid)
                )

    def test_trim_behavior(self):
        shape = (4,)
        offset = (0,)
        step = (1.0,)
        grid = grids.Grid(shape, step)

        base = tensor([11, 12, 13, 14], dtype=torch.float32)

        for width, expected, expected_offset in [
            (-1, tensor([12, 13, 14]), (1,)),
            (0, base, offset),
            (1, tensor([11, 12, 13]), (0,)),
            (2, tensor([11, 12]), (0,)),
        ]:
            with self.subTest(width=width):
                array = grids.GridVariable(base, offset, grid)
                trimmed = grids.trim(array, width, dim=0)
                self.assertArrayEqual(
                    trimmed, grids.GridVariable(expected, expected_offset, grid)
                )

    @parameterized.parameters(
        # Periodic BC
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=-2,
            expected_data=tensor([13, 14, 11, 12]),
            expected_offset=(-2,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=-1,
            expected_data=tensor([14, 11, 12, 13]),
            expected_offset=(-1,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=0,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 11]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=((BCType.PERIODIC, BCType.PERIODIC),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=2,
            expected_data=tensor([13, 14, 11, 12]),
            expected_offset=(2,),
        ),
        # Dirichlet BC
        dict(
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            input_data=tensor([11, 12, 13, 0]),
            input_offset=(1,),
            shift_offset=-1,
            expected_data=tensor([0, 11, 12, 13]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            input_data=tensor([0, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=0,
            expected_data=tensor([0, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=((BCType.DIRICHLET, BCType.DIRICHLET),),
            input_data=tensor([0, 12, 13, 14]),
            input_offset=(0,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 0]),
            expected_offset=(1,),
        ),
        # Neumann BC
        dict(
            bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            shift_offset=-1,
            expected_data=tensor([11, 11, 12, 13]),
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            shift_offset=0,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=((BCType.NEUMANN, BCType.NEUMANN),),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 14]),
            expected_offset=(1.5,),
        ),
        # Dirichlet / Neumann BC
        dict(
            bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
            input_data=tensor([0, 12, 13, 14, 15]),
            input_offset=(0.5,),
            shift_offset=-1,
            expected_data=tensor([0, 0, 12, 13, 14]),  # this is cell center
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=((BCType.DIRICHLET, BCType.NEUMANN),),
            input_data=tensor([0, 12, 13, 14, 15]),
            input_offset=(0.5,),
            shift_offset=1,
            expected_data=tensor([12, 13, 14, 15, 15]),
            expected_offset=(1.5,),
        ),
    )
    def test_shift_1d(
        self,
        bc_types,
        input_data,
        input_offset,
        shift_offset,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid(input_data.shape)
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.HomogeneousBoundaryConditions(bc_types)
        actual = bc.shift(array, shift_offset, dim=0)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([-13, -12, -11, 1, 12, 13, 14]),
            input_offset=(-3,),
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([1, 12, 13, 14]),
            input_offset=(0,),
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([1, 12, 13, 14, 2, -12, -11]),
            input_offset=(0,),
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([10, 11, 12, 13, 14]),
            input_offset=(-0.5,),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([12, 13, 14, 15, 13]),
            input_offset=(0.5,),
            expected_data=tensor([12, 13, 14, 15]),
            expected_offset=(0.5,),
        ),
        # Dirichlet / Neumann BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([-11, 12, 13, 14, 12]),
            input_offset=(-0.5,),
            expected_data=tensor([12, 13, 14, 12]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            grid_shape=4,
            input_data=tensor([12, 13, 14, 12]),
            input_offset=(0.5,),
            expected_data=tensor([12, 13, 14, 12]),
            expected_offset=(0.5,),
        ),
        # Periodic BC
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            grid_shape=4,
            input_data=tensor([-12, 11, 12, 13, 14]),
            input_offset=(-1,),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            grid_shape=4,
            input_data=tensor([11, 12, 13, 14, 12]),
            input_offset=(0,),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_trim_padding_1d(
        self,
        grid_shape,
        input_data,
        input_offset,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_shape,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual, _ = bc._trim_padding(array, dim=0)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        # test_pad_1d_inhomogeneous0
        dict(
          bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
          input_data=tensor([1, 12, 13, 14]),
          input_offset=(0,),
          width=-3,
          expected_data=tensor([-12, -11, -10, 1, 12, 13, 14]),
          expected_offset=(-3,),
      ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([1, 12, 13, 14]),
            input_offset=(0,),
            width=0,
            expected_data=tensor([1, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([1, 12, 13, 14]),
            input_offset=(0,),
            width=1,
            expected_data=tensor([1, 12, 13, 14, 2]),
            expected_offset=(0,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=-1,
            expected_data=tensor([12, 11, 12, 13, 14]),
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=0,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=1,
            expected_data=tensor([11, 12, 13, 14, 16]),
            expected_offset=(0.5,),
        ),
        # Dirichlet / Neumann BC, best test: test_pad_1d_inhomogeneous6
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=-1,
            expected_data=tensor([-9, 11, 12, 13, 14]),
            expected_offset=(-0.5,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=(2, 1),
            expected_data=tensor([-10, -9, 11, 12, 13, 14, 16]),
            expected_offset=(-1.5,),
        ),
    )
    def test_pad_1d_inhomogeneous(
        self, bc_types, input_data, input_offset, width, expected_data, expected_offset
    ):
        grid = grids.Grid((4,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual = grids.pad(array, width, dim=0, bc=bc)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            width=-1,
            axis=0,
            expected_data=tensor(
                [
                    [-11, -12, -13, -14],
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            expected_offset=(-0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            width=1,
            axis=1,
            expected_data=tensor(
                [
                    [11, 12, 13, 14, -14],
                    [21, 22, 23, 24, -24],
                    [31, 32, 33, 34, -34],
                ]
            ),
            expected_offset=(0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 0],
                    [21, 22, 23, 0],
                    [31, 32, 33, 0],
                ]
            ),
            input_offset=(0.5, 1),  # edge aligned offset
            width=-1,
            axis=1,
            expected_data=tensor(
                [
                    [0, 11, 12, 13, 0],
                    [0, 21, 22, 23, 0],
                    [0, 31, 32, 33, 0],
                ]
            ),
            expected_offset=(0.5, 0),
        ),
    )
    def test_pad_2d_dirichlet_cell_center(
        self, input_data, input_offset, width, axis, expected_data, expected_offset
    ):
        grid = grids.Grid(input_data.shape)
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim)
        actual = grids.pad(array, width, axis, bc=bc)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            values=((1.0, 2.0), (3.0, 4.0)),
            width=-1,
            axis=0,
            expected_data=tensor(
                [
                    [-9, -10, -11, -12],
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            expected_offset=(-0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 0.5),
            values=((1.0, 2.0), (3.0, 4.0)),
            width=1,
            axis=1,
            expected_data=tensor(
                [
                    [11, 12, 13, 14, -6],
                    [21, 22, 23, 24, -16],
                    [31, 32, 33, 34, -26],
                ]
            ),
            expected_offset=(0.5, 0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 4],
                    [21, 22, 23, 4],
                    [31, 32, 33, 4],
                ]
            ),
            input_offset=(0.5, 1),  # edge aligned offset
            values=((1.0, 2.0), (3.0, 4.0)),
            width=-1,
            axis=1,
            expected_data=tensor(
                [
                    [3, 11, 12, 13, 4],
                    [3, 21, 22, 23, 4],
                    [3, 31, 32, 33, 4],
                ]
            ),
            expected_offset=(0.5, 0),
        ),
    )
    def test_pad_2d_dirichlet_cell_center_inhomogeneous(
        self,
        input_data,
        input_offset,
        values,
        width,
        axis,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid(input_data.shape)
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim, values)
        actual = grids.pad(array, width, axis, bc=bc)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)


class BoundaryConditionsImposingTest(test_utils.TestCase):
    @parameterized.parameters(
        # Dirichlet BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([-14, -13, -12, 11, 12, 13, 14]),
            input_offset=(-3,),
            grid_size=4,
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, 2, -12, -11]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, 2, -12, -11]),
            input_offset=(1,),
            grid_size=5,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(1,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, -12, -11]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0.5,),
        ),
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14, 12]),
            input_offset=(-0.5,),
            grid_size=4,
            expected_data=tensor([12, 13, 14, 12]),
            expected_offset=(0.5,),
        ),
        # Periodic BC
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            input_data=tensor([-12, 11, 12, 13, 14]),
            input_offset=(-1,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            input_data=tensor([11, 12, 13, 14, 12]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_trim_boundary_1d(
        self,
        input_data,
        input_offset,
        grid_size,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_size,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual = bc.trim_boundary(array)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([12, 13, 14]),
            input_offset=(1,),
            grid_size=4,
            expected_data=tensor([1, 12, 13, 14, 2]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(1,),
            grid_size=5,
            expected_data=tensor([1, 11, 12, 13, 14, 2]),
            expected_offset=(0,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([-9, 11, 12, 13, 14, -10]),
            expected_offset=(-0.5,),
        ),
        # Neumann BC
        dict(
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            expected_data=tensor([12, 11, 12, 13, 14, 16]),
            expected_offset=(-0.5,),
        ),
        # Periodic BC
        dict(
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_pad_and_impose_bc_1d(
        self,
        input_data,
        input_offset,
        grid_size,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_size,))
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual = bc.pad_and_impose_bc(array, expected_offset, mode=Padding.EXTEND)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        # Dirichlet BC
        dict(
            input_data=tensor([0, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            expected_data=tensor([1, 12, 13, 14, 2]),
            expected_offset=(0,),
        ),
        dict(
            input_data=tensor([11, 12, 13, 14, 11]),
            input_offset=(1,),
            grid_size=5,
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            expected_data=tensor([1, 11, 12, 13, 14, 2]),
            expected_offset=(0,),
        ),
        dict(
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            bc_types=(((BCType.DIRICHLET, BCType.DIRICHLET),), ((1.0, 2.0),)),
            expected_data=tensor([-9, 11, 12, 13, 14, -10]),
            expected_offset=(-0.5,),
        ),
        # Neumann BC
        dict(
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            grid_size=4,
            bc_types=(((BCType.NEUMANN, BCType.NEUMANN),), ((1.0, 2.0),)),
            expected_data=tensor([12, 11, 12, 13, 14, 16]),
            expected_offset=(-0.5,),
        ),
        # Periodic BC
        dict(
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0,),
            grid_size=4,
            bc_types=(((BCType.PERIODIC, BCType.PERIODIC),), (None,)),
            expected_data=tensor([11, 12, 13, 14]),
            expected_offset=(0,),
        ),
    )
    def test_impose_bc_1d(
        self,
        input_data,
        input_offset,
        grid_size,
        bc_types,
        expected_data,
        expected_offset,
    ):
        grid = grids.Grid((grid_size,))
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        array = grids.GridVariable(input_data, input_offset, grid, bc)
        actual = bc.impose_bc(array, mode=Padding.EXTEND)
        expected = grids.GridVariable(expected_data, expected_offset, grid, bc)
        self.assertArrayEqual(actual, expected)

    @parameterized.parameters(
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            offset=(1.0, 0.5),
            values=((1.0, 2.0), (3.0, 4.0)),
            expected_data=tensor(
                [
                    [5, 1, 1, 1, 1, 7],
                    [-5, 11, 12, 13, 14, -6],
                    [-15, 21, 22, 23, 24, -16],
                    [4, 2, 2, 2, 2, 6],
                ]
            ),
            expected_offset=(0.0, -0.5),
        ),
        dict(
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            offset=(0.5, 0.0),
            values=((1.0, 2.0), (3.0, 4.0)),
            expected_data=tensor(
                [
                    [3, -10, -11, -12, 4],
                    [3, 12, 13, 14, 4],
                    [3, 22, 23, 24, 4],
                    [3, 32, 33, 34, 4],
                    [3, -28, -29, -30, 4],
                ]
            ),
            expected_offset=(-0.5, 0.0),
        ),
    )
    def test_impose_bc_2d_constant_boundary(
        self, input_data, offset, values, expected_data, expected_offset
    ):
        grid = grids.Grid(input_data.shape)
        bc = boundaries.dirichlet_boundary_conditions(grid.ndim, values)
        variable = grids.GridVariable(input_data, offset, grid, bc)
        variable = variable.impose_bc(mode=Padding.EXTEND)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(variable, expected)

    @parameterized.parameters(
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=((2, 1),),
            expected_data=tensor([-10, -9, 11, 12, 13, 14, 16]),
            expected_offset=(-1.5,),
        ),
        dict(
            bc_types=(((BCType.DIRICHLET, BCType.NEUMANN),), ((1.0, 2.0),)),
            input_data=tensor([11, 12, 13, 14]),
            input_offset=(0.5,),
            width=((2, 0),),
            expected_data=tensor([-10, -9, 11, 12, 13, 14]),
            expected_offset=(-1.5,),
        ),
        dict(
            bc_types=(
                (
                    (BCType.PERIODIC, BCType.PERIODIC),
                    (BCType.DIRICHLET, BCType.DIRICHLET),
                ),
                ((0.0, 0.0), (0.0, 0.0)),
            ),
            input_data=tensor(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                ]
            ),
            input_offset=(0.5, 1),
            width=((1, 1), (1, 1)),
            expected_data=tensor(
                [
                    [0, 31, 32, 33, 0, -33],
                    [0, 11, 12, 13, 0, -13],
                    [0, 21, 22, 23, 0, -23],
                    [0, 31, 32, 33, 0, -33],
                    [0, 11, 12, 13, 0, -13],
                ]
            ),
            expected_offset=(-0.5, 0.0),
        ),
    )
    def test_pad_all(
        self, bc_types, input_data, input_offset, width, expected_data, expected_offset
    ):
        grid = grids.Grid(input_data.shape)
        array = grids.GridVariable(input_data, input_offset, grid)
        bc = boundaries.ConstantBoundaryConditions(bc_types[0], bc_types[1])
        actual = bc.pad_all(array, width, mode=boundaries.Padding.MIRROR)
        expected = grids.GridVariable(expected_data, expected_offset, grid)
        self.assertArrayEqual(actual, expected)


class PressureBoundaryConditionsTest(test_utils.TestCase):
    def test_get_pressure_bc_from_velocity_2d(self):
        grid = grids.Grid((10, 10))
        bc = boundaries.dirichlet_boundary_conditions(ndim=2)
        u_array = grids.GridVariable(torch.zeros(grid.shape), (1, 0.5), grid, bc)
        v_array = grids.GridVariable(torch.zeros(grid.shape), (0.5, 1), grid, bc)
        v = grids.GridVariableVector(tuple([u_array, v_array]))
        pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
        self.assertEqual(
            pressure_bc.types,
            ((BCType.NEUMANN, BCType.NEUMANN), (BCType.NEUMANN, BCType.NEUMANN)),
        )


class VariableBoundaryConditionsTest(test_utils.TestCase):
    """DiscreteBoundaryConditions and FunctionBoundaryConditions."""

    def test_initialization_with_arrays(self):
        """Test initialization with tensor boundary values."""
        grid = grids.Grid((4, 4))
        
        # Create boundary value arrays
        left_boundary = tensor([1., 2., 3., 4.])   # left boundary (4 points)
        right_boundary = tensor([5., 6., 7., 8.])  # right boundary (4 points)
        bottom_boundary = tensor([10., 11., 12., 13.])  # bottom boundary (4 points)
        top_boundary = tensor([20., 21., 22., 23.])     # top boundary (4 points)
        
        bc = boundaries.DiscreteBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=((left_boundary, right_boundary),
                    (bottom_boundary, top_boundary))
        )
        
        self.assertEqual(len(bc.bc_values), 2)
        self.assertEqual(len(bc.bc_values[0]), 2)
        self.assertEqual(len(bc.bc_values[1]), 2)

    def test_initialization_with_callables(self):
        """Test initialization with callable boundary values."""
        def inlet_profile(x, y):
            return torch.sin(torch.pi * y)
        
        def outlet_profile(x, y):
            return torch.zeros_like(y)
        
        grid = grids.Grid((4, 4))

        bc = boundaries.FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.PERIODIC, BCType.PERIODIC)),
            values=((inlet_profile, outlet_profile),
                    (None, None)),
            grid=grid,
            offset=(0.5, 0.5)
        )
        
        # raw is the callable, bc_values is the evaluated tensor
        self.assertTrue(callable(bc._raw_bc_values[0][0]))
        self.assertTrue(callable(bc._raw_bc_values[0][1]))
        self.assertTrue(isinstance(bc._bc_values[0][0], torch.Tensor))
        self.assertTrue(isinstance(bc._bc_values[0][1], torch.Tensor))

    def test_initialization_with_mixed_values(self):
        """Test initialization with mixed boundary value types."""
        left_boundary = tensor([1., 2., 3., 4.])
        
        def right_profile(x, y):
            return y * 2.0
        
        bc = boundaries.FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.NEUMANN, BCType.NEUMANN)),
            values=((left_boundary, right_profile),
                    (0.0, 1.0)),  # constant values
            grid=grids.Grid((4, 4)),
            offset=(0.5, 0.5)
        )
        
        # Use private _bc_values attribute to check types without grid context
        self.assertIsInstance(bc._bc_values[0][0], torch.Tensor)
        self.assertTrue(callable(bc._raw_bc_values[0][1]))
        self.assertTrue(isinstance(bc._bc_values[0][1], torch.Tensor))
        self.assertEqual(bc._bc_values[1][0], 0.0)
        self.assertEqual(bc._bc_values[1][1], 1.0)

    def test_evaluate_boundary_value_with_callable(self):
        """Test evaluation of callable boundary values."""
        def parabolic_profile(x, y):
            return y * (1 - y)
        
        grid = grids.Grid((4, 4), domain=((0, 1), (0, 1)))
        expected = tensor([0., 0.1875, 0.25, 0.1875])
        
        bc = boundaries.FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.NEUMANN, BCType.NEUMANN)),
            values=((parabolic_profile, 0.0),
                    (0.0, 1.0)),  # constant values
            grid=grid,
            offset=(0.0, 0.0)
        )
        
        result = bc.bc_values[0][0]  # Evaluate left boundary
        self.assertArrayEqual(result, expected)

    def test_get_boundary_coordinates_2d(self):
        """Test boundary coordinate generation for 2D case."""
        grid = grids.Grid((4, 4), domain=((0, 2), (0, 2)))
        offset = (0.5, 0.5)
        
        bc = boundaries.DiscreteBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=((1.0, 2.0), (3.0, 4.0))
        )
        
        # Test boundary coordinates for dimension 0 (y-coordinates for left/right boundaries)
        coords = bc._boundary_mesh(0, grid, offset)
        expected_x = tensor([0, 0, 0, 0])  # x-coordinates for left boundary
        expected_y = tensor([0.25, 0.75, 1.25, 1.75])  # offset[1] * step[1] for each grid point
        # left edge
        self.assertArrayEqual(coords[0][0], expected_x)        
        self.assertArrayEqual(coords[0][1], expected_y)
        # right edge
        expected_x = tensor([2, 2, 2, 2])  #
        self.assertArrayEqual(coords[1][0], expected_x)
        self.assertArrayEqual(coords[1][1], expected_y)

        
        # Test boundary coordinates for dimension 1 (x-coordinates for top/bottom boundaries)
        coords = bc._boundary_mesh(1, grid, offset)
        expected_x = tensor([0.25, 0.75, 1.25, 1.75])  # offset[0] * step[0] for each grid point
        expected_y = tensor([0, 0, 0, 0]) # y-coordinates for bottom boundary
        # bottom edge
        self.assertArrayEqual(coords[0][0], expected_x)
        self.assertArrayEqual(coords[0][1], expected_y)
        # top edge
        expected_y = tensor([2, 2, 2, 2])  #
        self.assertArrayEqual(coords[1][0], expected_x)
        self.assertArrayEqual(coords[1][1], expected_y)

    def test_values_method_with_callables(self):
        """Test values method with callable boundary conditions."""
        grid = grids.Grid((3, 3), domain=((0, 3), (0, 3)))
        
        def left_profile(x, y):
            return y * 2.0
        
        def right_profile(x, y):
            return y + 1.0
        
        bc = boundaries.FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.PERIODIC, BCType.PERIODIC)),
            values=((left_profile, right_profile),
                    (None, None)),
            grid=grid,
            offset=(0.0, 0.5)
        )
        
        bc_lower, bc_upper = bc.bc_values[0]
        
        # For the default edge center (0.0, 0.5), y coordinates are [0.5, 1.5, 2.5]
        expected_left = tensor([1.0, 3.0, 5.0])   # y * 2.0
        expected_right = tensor([1.5, 2.5, 3.5])  # y + 1.0
        
        self.assertArrayEqual(bc_lower, expected_left)
        self.assertArrayEqual(bc_upper, expected_right)


    def test_variable_boundary_vs_constant_boundary_consistency(self):
        """Test that VariableBoundaryConditions gives same results as ConstantBoundaryConditions for constant values."""
        grid = grids.Grid((8, 4))
        
        # Constant boundary conditions
        const_bc = boundaries.ConstantBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=((1.0, 2.0), (3.0, 4.0))
        )
        
        # Variable boundary conditions with constant values
        var_bc = boundaries.DiscreteBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=((torch.ones(4), 2*torch.ones(4)), 
                    (3*torch.ones(8), 4*torch.ones(8))
        ))
        
        # Test that values method gives same results
        const_values_0 = const_bc.values(0, grid)
        var_values_0 = var_bc.values(0, grid)
        
        # Both should give scalar constant values broadcast to boundary shape
        expected_left = torch.full((4,), 1.0)
        expected_right = torch.full((4,), 2.0)
        self.assertArrayEqual(const_values_0[0], expected_left)
        self.assertArrayEqual(var_values_0[0], expected_left)
        self.assertArrayEqual(const_values_0[1], expected_right)
        self.assertArrayEqual(var_values_0[1], expected_right)


    def test_compatible_bc_with_grid(self):
        """Test error handling in VariableBoundaryConditions."""
        # Test with mismatched array dimensions
        grid = grids.Grid((4, 4))
        wrong_size_boundary = tensor([1., 2.])  # Should be size 4
        
        bc = boundaries.DiscreteBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=((wrong_size_boundary, 2.0),
                    (3.0, 4.0))
        )
        
        # This should raise an error when validating with grid
        with self.assertRaises(ValueError):
            bc._validate_boundary_arrays_with_grid(grid)

    @parameterized.named_parameters(
        # Test different offsets for linear function
        dict(
            testcase_name="_linear_lower_left_corner",
            shape=(4, 4),
            offset=(0, 0),
            f=lambda x, y: x + 2 * y,
            expected_bc_values=(
                (tensor([0., 0.5, 1.0, 1.5]), tensor([1., 1.5, 2.0, 2.5])),
                (tensor([0., 0.25, 0.5, 0.75]), tensor([2., 2.25, 2.5, 2.75]))
            ),
        ),
        dict(
            testcase_name="_linear_cell_center",
            shape=(4, 4),
            offset=(0.5, 0.5),
            f=lambda x, y: x + 2 * y,
            expected_bc_values=(
                (tensor([0.25, 0.75, 1.25, 1.75]), tensor([1.25, 1.75, 2.25, 2.75])),
                (tensor([0.125, 0.375, 0.625, 0.875]), tensor([2.125, 2.375, 2.625, 2.875]))
            ),
        ),
        dict(
            testcase_name="_linear_upper_right_corner",
            shape=(4, 4),
            offset=(1.0, 1.0),
            f=lambda x, y: x + 2 * y,
            expected_bc_values=(
                (tensor([0.5, 1.0, 1.5, 2.0]), tensor([1.5, 2.0, 2.5, 3.0])),
                (tensor([0.25, 0.5, 0.75, 1.0]), tensor([2.25, 2.5, 2.75, 3.0]))
            ),
        ),
        # Test constant function
        dict(
            testcase_name="_constant_cell_center",
            shape=(4, 4),
            offset=(0.5, 0.5),
            f=lambda x, y: torch.ones_like(x),
            expected_bc_values=(
                (tensor([1., 1., 1., 1.]), tensor([1., 1., 1., 1.])),
                (tensor([1., 1., 1., 1.]), tensor([1., 1., 1., 1.]))
            ),
        ),
        # Test polynomial function
        dict(
            testcase_name="_quadratic_lower_left_corner",
            shape=(3, 3),
            offset=(0, 0),
            f=lambda x, y: x**2 + y**2,
            expected_bc_values=(
                (tensor([0., 1./9., 4./9.]), tensor([1., 10./9., 13./9.])),
                (tensor([0., 1./9., 4./9.]), tensor([1., 10./9., 13./9.]))
            ),
        ),
    )
    def test_function_bc_evaluation(self, shape, offset, f, expected_bc_values):
        """Test that FunctionBoundaryConditions evaluates functions correctly."""
        grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
        
        bc = boundaries.FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=((f, f), (f, f)),
            grid=grid,
            offset=offset
        )
        
        actual_bc_values = bc.bc_values
        self.assertNestedTuplesEqual(actual_bc_values, expected_bc_values)

    @parameterized.named_parameters(
        dict(
            testcase_name="_linear",
            shape=(16, 16),
            f=lambda x, y: x + 2 * y,
        ),
        dict(
            testcase_name="_quadratic",
            shape=(16, 16),
            f=lambda x, y: x**2 + 2 * y**2,
        ),
    )
    def test_function_bc_vs_discrete_bc_consistency(self, shape, f):
        """Test that FunctionBoundaryConditions gives same results as DiscreteBoundaryConditions for equivalent inputs."""
        # Create a function that matches predefined tensor values
        
        grid = grids.Grid(shape, domain=((0.0, 1.0), (0.0, 1.0)))
        offsets = [(0, 0), (0, 1), (1, 0), (0.5, 1), (1, 0.5), (1, 1)]
        
        for offset in offsets:
            # Pre-calculate what the function should give
            lower_coords, upper_coords = grid.boundary_mesh(0, offset)
            expected_left = f(*lower_coords)
            expected_right = f(*upper_coords)
            
            lower_coords, upper_coords = grid.boundary_mesh(1, offset)
            expected_bottom = f(*lower_coords)
            expected_top = f(*upper_coords)
            
            # Function-based BC
            function_bc = boundaries.FunctionBoundaryConditions(
                types=((BCType.DIRICHLET, BCType.DIRICHLET),
                    (BCType.DIRICHLET, BCType.DIRICHLET)),
                values=f,
                grid=grid,
                offset=offset
            )
            
            # Discrete BC with pre-calculated values
            discrete_bc = boundaries.DiscreteBoundaryConditions(
                types=((BCType.DIRICHLET, BCType.DIRICHLET),
                    (BCType.DIRICHLET, BCType.DIRICHLET)),
                values=((expected_left, expected_right),
                        (expected_bottom, expected_top))
            )
            
            # Compare bc_values
            self.assertNestedTuplesEqual(function_bc.bc_values, discrete_bc.bc_values, atol=1e-6)

    def test_function_bc_time_dependent(self):
        """Test FunctionBoundaryConditions with time-dependent functions."""
        def time_varying_inlet(x, y, t):
            return torch.sin(2 * torch.pi * t) * y
        
        def steady_outlet(x, y, t):
            return torch.zeros_like(y)
        
        grid = grids.Grid((3, 3), domain=((0.0, 1.0), (0.0, 1.0)))
        time = torch.tensor([0.25])  # sin(2*pi*0.25) = sin(pi/2) = 1
        
        bc = boundaries.FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.PERIODIC, BCType.PERIODIC)),
            values=((time_varying_inlet, steady_outlet),
                    (None, None)),
            grid=grid,
            offset=(0.5, 0.5),
            time=time
        )
        
        bc_values = bc.bc_values
        
        # At t=0.25, sin(2*pi*0.25) = 1, so inlet should be just y
        expected_inlet = tensor([1./6., 0.5, 5./6.])  # y values at offset (0.5, 0.5)
        expected_outlet = tensor([0., 0., 0.])
        
        self.assertAllClose(bc_values[0][0], expected_inlet, atol=1e-6, rtol=1e-8)
        self.assertAllClose(bc_values[0][1], expected_outlet, atol=1e-6, rtol=1e-8)


    def test_function_bc_mixed_values(self):
        """Test FunctionBoundaryConditions with mixed value types."""
        def parabolic_profile(x, y):
            return y * (1 - y)
        
        grid = grids.Grid((4, 4), domain=((0.0, 1.0), (0.0, 1.0)))
        left_boundary = tensor([0.1, 0.2, 0.3, 0.4])
        
        bc = boundaries.FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.NEUMANN, BCType.NEUMANN)),
            values=((left_boundary, parabolic_profile),
                    (0.5, 1.0)),  # constant values
            grid=grid,
            offset=(0.0, 0.0)
        )
        
        bc_values = bc.bc_values
        
        # Check left boundary (tensor input)
        self.assertAllClose(bc_values[0][0], left_boundary)
        
        # Check right boundary (function evaluated)
        expected_right = tensor([0., 3./16., 1./4., 3./16.])  # y*(1-y) at y=[0, 0.25, 0.5, 0.75]
        self.assertAllClose(bc_values[0][1], expected_right, atol=1e-6, rtol=1e-8)
        
        # Check constant boundaries
        self.assertEqual(bc_values[1][0], tensor(0.5))
        self.assertEqual(bc_values[1][1], tensor(1.0))



if __name__ == "__main__":
    absltest.main()