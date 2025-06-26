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


import dataclasses
import math
import numbers
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from torch_cfd import grids

BoundaryConditions = grids.BoundaryConditions
Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


BCType = grids.BCType()
Padding = grids.Padding()
BCValue = grids.BCValue


@dataclasses.dataclass(init=False, frozen=True, repr=False)
class ConstantBoundaryConditions(BoundaryConditions):
    """Boundary conditions for a PDE variable that are constant in space and time.

    Attributes:
        types: a tuple of tuples, where `types[i]` is a tuple specifying the lower and upper BC types for dimension `i`. The types can be one of the following:
            BCType.PERIODIC, BCType.DIRICHLET, BCType.NEUMANN.
        values: a tuple of tuples, where `values[i]` is a tuple specifying the lower and upper boundary values for dimension `i`. If None, the boundary condition is homogeneous (zero).

    Example usage:
      grid = Grid((10, 10))
      bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)),
                                          ((0.0, 10.0),(1.0, 0.0)))
      # in dimension 0 is periodic, (0, 10) on left and right (un-used)
      # in dimension 1 is dirichlet, (1, 0) on bottom and top.
      v = GridVariable(torch.zeros((10, 10)), offset=(0.5, 0.5), grid, bc)
      # v.offset is (0.5, 0.5) which is the cell center, so the boundary conditions have no effect in this case

    """

    _types: Tuple[Tuple[str, str], ...]
    bc_values: Tuple[Tuple[Optional[float], Optional[float]], ...]
    ndim: int

    def __init__(
        self,
        types: Sequence[Tuple[str, str]],
        values: Sequence[Tuple[Optional[float], Optional[float]]],
    ):
        types = tuple(types)
        values = tuple(values)
        object.__setattr__(self, "_types", types)
        object.__setattr__(self, "bc_values", values)
        object.__setattr__(self, "ndim", len(types))

    @property
    def types(self) -> Tuple[Tuple[str, str], ...]:
        """Returns the boundary condition types."""
        return self._types

    @types.setter
    def types(self, bc_types: Sequence[Tuple[str, str]]) -> None:
        """Sets the boundary condition types and updates ndim accordingly."""
        bc_types = tuple(bc_types)
        assert self.ndim == len(
            bc_types
        ), f"Number of dimensions {self.ndim} does not match the number of types {bc_types}."
        object.__setattr__(self, "_types", bc_types)

    def __repr__(self) -> str:
        try:
            lines = [f"{self.__class__.__name__}({self.ndim}D):"]

            for dim in range(self.ndim):
                lower_type, upper_type = self.types[dim]
                lower_val, upper_val = self.bc_values[dim]

                # Format values
                lower_val_str = "None" if lower_val is None else f"{lower_val}"
                upper_val_str = "None" if upper_val is None else f"{upper_val}"

                lines.append(
                    f"  dim {dim}: [{lower_type}({lower_val_str}), {upper_type}({upper_val_str})]"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"{self.__class__.__name__} not initialized: {e}"

    def clone(
        self,
        types: Optional[Sequence[Tuple[str, str]]] = None,
        values: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    ) -> BoundaryConditions:
        """Creates a copy of this boundary condition, optionally with modified parameters.

        Args:
            types: New boundary condition types. If None, uses current types.
            values: New boundary condition values. If None, uses current values.

        Returns:
            A new ConstantBoundaryConditions instance.
        """
        new_types = types if types is not None else self.types
        new_values = values if values is not None else self.bc_values
        return ConstantBoundaryConditions(new_types, new_values)

    def shift(
        self,
        u: GridVariable,
        offset: int,
        dim: int,
    ) -> GridVariable:
        """
        A fallback function to make the implementation back-compatible
        see grids.shift
        bc.shift(u, offset, dim) overrides u.bc
        grids.shift(u, offset, dim) keeps u.bc
        """
        return grids.shift(u, offset, dim, self)

    def _is_aligned(self, u: GridVariable, dim: int) -> bool:
        """Checks if array u contains all interior domain information.

        For dirichlet edge aligned boundary, the value that lies exactly on the
        boundary does not have to be specified by u.
        Neumann edge aligned boundary is not defined.

        Args:
        u: GridVariable that should contain interior data
        dim: axis along which to check

        Returns:
        True if u is aligned, and raises error otherwise.
        """
        size_diff = u.shape[dim] - u.grid.shape[dim]
        if self.types[dim][0] == BCType.DIRICHLET and math.isclose(u.offset[dim], 1):
            size_diff += 1
        if self.types[dim][1] == BCType.DIRICHLET and math.isclose(u.offset[dim], 1):
            size_diff += 1
        if (
            self.types[dim][0] == BCType.NEUMANN and math.isclose(u.offset[dim], 1)
        ) or (self.types[dim][1] == BCType.NEUMANN and math.isclose(u.offset[dim], 0)):
            """
            if lower (or left for dim 0) is Neumann, and the offset is 1 (the variable is on the right edge of a cell), the Neumann bc is not defined; vice versa for upper (or right for dim 0) Neumann bc with offset 0 (the variable is on the left edge of a cell).
            """
            raise ValueError("Variable not aligned with Neumann BC")
        if size_diff < 0:
            raise ValueError(
                "the GridVariable does not contain all interior grid values."
            )
        return True

    def pad(
        self,
        u: GridVariable,
        width: Union[Tuple[int, int], int],
        dim: int,
        mode: Optional[str] = Padding.EXTEND,
    ) -> GridVariable:
        """Wrapper for grids.pad with a specific bc.

        Args:
          u: a `GridVariable` object.
          width: number of elements to pad along axis. If width is an int, use
            negative value for lower boundary or positive value for upper boundary.
            If a tuple, pads with width[0] on the left and width[1] on the right.
          dim: axis to pad along.
          mode: type of padding to use in non-periodic case.
            Mirror mirrors the array values across the boundary.
            Extend extends the last well-defined array value past the boundary.

        Returns:
          Padded array, elongated along the indicated axis.
          the original u.bc will be replaced with self.
        """
        _ = self._is_aligned(u, dim)
        if isinstance(width, tuple) and (width[0] > 0 and width[1] > 0):
            need_trimming = "both"
        elif (isinstance(width, tuple) and (width[0] > 0 and width[1] == 0)) or (
            isinstance(width, int) and width < 0
        ):
            need_trimming = "left"
        elif (isinstance(width, tuple) and (width[0] == 0 and width[1] > 0)) or (
            isinstance(width, int) and width > 0
        ):
            need_trimming = "right"
        else:
            need_trimming = "none"

        u, trimmed_padding = self._trim_padding(u, dim, need_trimming)

        if isinstance(width, int):
            if width < 0:
                width -= trimmed_padding[0]
            if width > 0:
                width += trimmed_padding[1]
        elif isinstance(width, tuple):
            width = (width[0] + trimmed_padding[0], width[1] + trimmed_padding[1])

        u = grids.pad(u, width, dim, self, mode=mode)
        return u

    def pad_all(
        self,
        u: GridVariable,
        width: Tuple[Tuple[int, int], ...],
        mode: Optional[str] = Padding.EXTEND,
    ) -> GridVariable:
        """Pads along all axes with pad width specified by width tuple.

        Args:
          u: a `GridVariable` object.
          width: Tuple of padding width for each side for each axis.
          mode: type of padding to use in non-periodic case.
            Mirror mirrors the array values across the boundary.
            Extend extends the last well-defined array value past the boundary.

        Returns:
          Padded array, elongated along all axes.
        """
        for dim in range(-u.grid.ndim, 0):
            u = self.pad(u, width[dim], dim, mode=mode)
        return u

    def values(
        self, dim: int, grid: Grid
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns boundary values on the grid along axis.

        Args:
          dim: axis along which to return boundary values.
          grid: a `Grid` object on which to evaluate boundary conditions.

        Returns:
          A tuple of arrays of grid.ndim - 1 dimensions that specify values on the
          boundary. In case of periodic boundaries, returns a tuple(None,None).
        """
        if None in self.bc_values[dim]:
            return (None, None)

        bc = []
        for i in [0, 1]:
            value = self.bc_values[dim][-i]
            if value is None:
                bc.append(None)
            elif isinstance(value, float):
                bc.append(torch.full(grid.shape[:dim] + grid.shape[dim + 1 :], value))
            elif isinstance(value, torch.Tensor):
                if value.shape != grid.shape[:dim] + grid.shape[dim + 1 :]:
                    raise ValueError(
                        f"Boundary value shape {value.shape} does not match expected shape {grid.shape[:dim] + grid.shape[dim + 1 :]}"
                    )
                bc.append(value)

        return tuple(bc)

    def _trim_padding(
        self, u: GridVariable, dim: int = -1, trim_side: str = "both"
    ) -> Tuple[GridVariable, Tuple[int, int]]:
        """Trims padding from a GridVariable along axis and returns the array interior.

        Args:
        u: a `GridVariable` object.
        dim: axis to trim along.
        trim_side: if 'both', trims both sides. If 'right', trims the right side.
            If 'left', the left side.

        Returns:
        Trimmed array, shrunk along the indicated axis side. bc is updated to None
        """
        positive_trim = 0
        negative_trim = 0
        padding = (0, 0)

        if trim_side not in ("both", "left", "right"):
            return u.array, padding

        if u.shape[dim] >= u.grid.shape[dim]:
            # number of cells that were padded on the left
            negative_trim = 0
            if u.offset[dim] <= 0 and (trim_side == "both" or trim_side == "left"):
                negative_trim = -math.ceil(-u.offset[dim])
                # periodic is a special case. Shifted data might still contain all the
                # information.
                if self.types[dim][0] == BCType.PERIODIC:
                    negative_trim = max(negative_trim, u.grid.shape[dim] - u.shape[dim])
                # for both DIRICHLET and NEUMANN cases the value on grid.domain[0] is
                # a dependent value.
                elif math.isclose(u.offset[dim] % 1, 0):
                    negative_trim -= 1
                u = grids.trim(u, negative_trim, dim)
            # number of cells that were padded on the right
            positive_trim = 0
            if trim_side == "right" or trim_side == "both":
                # periodic is a special case. Boundary on one side depends on the other
                # side.
                if self.types[dim][1] == BCType.PERIODIC:
                    positive_trim = max(u.shape[dim] - u.grid.shape[dim], 0)
                else:
                    # for other cases, where to trim depends only on the boundary type
                    # and data offset.
                    last_u_offset = u.shape[dim] + u.offset[dim] - 1
                    boundary_offset = u.grid.shape[dim]
                    if last_u_offset >= boundary_offset:
                        positive_trim = math.ceil(last_u_offset - boundary_offset)
                        if self.types[dim][1] == BCType.DIRICHLET and math.isclose(
                            u.offset[dim] % 1, 0
                        ):
                            positive_trim += 1
        if positive_trim > 0:
            u = grids.trim(u, positive_trim, dim)
        # combining existing padding with new padding
        padding = (-negative_trim, positive_trim)
        return u, padding

    def trim_boundary(self, u: GridVariable) -> GridVariable:
        """Returns GridVariable without the grid points on the boundary.

        Some grid points of GridVariable might coincide with boundary. This trims those values.
        If the array was padded beforehand, removes the padding.

        Args:
        u: a `GridVariable` object.

        Returns:
        A GridVariable shrunk along certain dimensions.
        """
        for dim in range(-u.grid.ndim, 0):
            _ = self._is_aligned(u, dim)
            u, _ = self._trim_padding(u, dim)
        return GridVariable(u.data, u.offset, u.grid)

    def pad_and_impose_bc(
        self,
        u: GridVariable,
        offset_to_pad_to: Optional[Tuple[float, ...]] = None,
        mode: Optional[str] = "",
    ) -> GridVariable:
        """Returns GridVariable with correct boundary values.

        Some grid points of GridVariable might coincide with boundary, thus this function is only used with the trimmed GridVariable.
        Args:
            - u: a `GridVariable` object that specifies only scalar values on the internal nodes.
            - offset_to_pad_to: a Tuple of desired offset to pad to. Note that if the function is given just an interior array in dirichlet case, it can pad to both 0 offset and 1 offset.
            - mode: type of padding to use in non-periodic case. Mirror mirrors the flow across the boundary. Extend extends the last well-defined value past the boundary. None means no ghost cell padding.

        Returns:
        A GridVariable that has correct boundary values.
        """
        assert u.bc is None, "u must be trimmed before padding and imposing bc."
        if offset_to_pad_to is None:
            offset_to_pad_to = u.offset
        for dim in range(-u.grid.ndim, 0):
            _ = self._is_aligned(u, dim)
            if self.types[dim][0] != BCType.PERIODIC:
                if mode:
                    # if the offset is either 0 or 1, u is aligned with the boundary and is defined on cell edges on one side of the boundary, if trim_boundary is called before this function.
                    # u needs to be padded on both sides
                    # if the offset is 0.5, one ghost cell is needed on each side.
                    # it will be taken care by grids.pad function automatically.
                    u = grids.pad(u, (1, 1), dim, self, mode=mode)
                elif self.types[dim][0] == BCType.DIRICHLET and not mode:
                    if self.types[dim][1] == BCType.DIRICHLET and math.isclose(offset_to_pad_to[dim], 1.0):
                        u = grids.pad(u, 1, dim, self)
                    elif math.isclose(offset_to_pad_to[dim], 0.0):
                        u = grids.pad(u, -1, dim, self)
        return GridVariable(u.data, u.offset, u.grid, self)

    def impose_bc(self, u: GridVariable, mode: str = "") -> GridVariable:
        """Returns GridVariable with correct boundary condition.

        Some grid points of GridVariable might coincide with boundary. This ensures
        that the GridVariable.array agrees with GridVariable.bc.
        Args:
        u: a `GridVariable` object.

        Returns:
        A GridVariable that has correct boundary values.

        Notes:
        If one needs ghost_cells, please use a manual function pad_all to add ghost cells are added on the other side of DoFs living at cell center if the bc is Dirichlet or Neumann.
        """
        offset = u.offset
        u = self.trim_boundary(u)
        u = self.pad_and_impose_bc(u, offset, mode)
        return u


class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
    """Boundary conditions for a PDE variable.

    Example usage:
      grid = Grid((10, 10))
      bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)))
      u = GridVariable(torch.zeros((10, 10)), offset=(0.5, 0.5), grid, bc)


    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """

    def __init__(self, types: Sequence[Tuple[str, str]]):

        ndim = len(types)
        values = ((0.0, 0.0),) * ndim
        super(HomogeneousBoundaryConditions, self).__init__(types, values)


def is_bc_periodic_boundary_conditions(bc: Optional[BoundaryConditions], dim: int) -> bool:
    """Returns true if scalar has periodic bc along axis."""
    if bc is None:
        return False
    elif bc.types[dim][0] != BCType.PERIODIC:
        return False
    elif bc.types[dim][0] == BCType.PERIODIC and bc.types[dim][0] != bc.types[dim][1]:
        raise ValueError(
            "periodic boundary conditions must be the same on both sides of the axis"
        )
    return True


def is_bc_all_periodic_boundary_conditions(bc: BoundaryConditions) -> bool:
    """Returns true if scalar has periodic bc along all axes."""
    for dim in range(bc.ndim):
        if not is_bc_periodic_boundary_conditions(bc, dim):
            return False
    return True


def is_periodic_boundary_conditions(c: GridVariable, dim: int) -> bool:
    """Returns true if scalar has periodic bc along axis."""
    return is_bc_periodic_boundary_conditions(c.bc, dim)


def is_bc_pure_neumann_boundary_conditions(bc: Optional[BoundaryConditions]) -> bool:
    """Returns true if scalar has pure Neumann bc along all axes."""
    if bc is None:
        return False
    for dim in range(bc.ndim):
        if bc.types[dim][0] != BCType.NEUMANN or bc.types[dim][1] != BCType.NEUMANN:
            return False
    return True


def is_pure_neumann_boundary_conditions(c: GridVariable) -> bool:
    """Returns true if scalar has pure Neumann bc along all axes."""
    return is_bc_pure_neumann_boundary_conditions(c.bc)


# Convenience utilities to ease updating of BoundaryConditions implementation
def periodic_boundary_conditions(ndim: int) -> BoundaryConditions:
    """Returns periodic BCs for a variable with `ndim` spatial dimension."""
    return HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),) * ndim)


def dirichlet_boundary_conditions(
    ndim: int,
    bc_values: Optional[Sequence[Tuple[float, float]]] = None,
) -> BoundaryConditions:
    """Returns Dirichelt BCs for a variable with `ndim` spatial dimension.

    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_values:
        return HomogeneousBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim
        )
    else:
        return ConstantBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim, bc_values
        )


def neumann_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> BoundaryConditions:
    """Returns Neumann BCs for a variable with `ndim` spatial dimension.

    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(((BCType.NEUMANN, BCType.NEUMANN),) * ndim)
    else:
        return ConstantBoundaryConditions(
            ((BCType.NEUMANN, BCType.NEUMANN),) * ndim, bc_vals
        )


def dirichlet_and_periodic_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
    periodic_dim: int = 1,
) -> ConstantBoundaryConditions:
    """Returns BCs Periodic for dimension i and Dirichlet for the complementary dimension.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if periodic_dim < 0:
        periodic_dim += 2

    dir_dim = periodic_dim ^ 1  # flip the bit to get the other dimension

    dir_bc = (BCType.DIRICHLET, BCType.DIRICHLET)
    periodic_bc = (BCType.PERIODIC, BCType.PERIODIC)

    # Cleaner approach: build the types list directly
    types = [periodic_bc if i == periodic_dim else dir_bc for i in range(2)]

    if not bc_vals:
        return HomogeneousBoundaryConditions(tuple(types))
    else:
        _bc_vals: list[tuple[Optional[float], Optional[float]]] = [
            (None, None),
            (None, None),
        ]
        # periodic dim gets 'no-op' values
        _bc_vals[periodic_dim] = (None, None)

        if len(bc_vals) != 2:
            raise ValueError(
                f"Expected bc_vals to be a tuple of length 2, got {len(bc_vals)}"
            )
        # the orthogonal (Dirichlet) dim gets the user-provided values
        _bc_vals[dir_dim] = bc_vals

        return ConstantBoundaryConditions(
            tuple(types),
            tuple(_bc_vals),
        )


def channel_flow_2d_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
) -> ConstantBoundaryConditions:
    """Create channel flow boundary conditions."""
    return dirichlet_and_periodic_boundary_conditions(bc_vals, periodic_dim=0)


def periodic_and_neumann_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
) -> BoundaryConditions:
    """Returns BCs periodic for dimension 0 and Neumann for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(
            ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN))
        )
    else:
        return ConstantBoundaryConditions(
            ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)),
            ((None, None), bc_vals),
        )


@dataclasses.dataclass(init=False, frozen=True, repr=False)
class DiscreteBoundaryConditions(ConstantBoundaryConditions):
    """Boundary conditions that can vary spatially along the boundary.

    Array-based values that are evaluated at boundary nodes with proper offsets. The values must match a variable's offset in order that the numerical differentiation is correct.

    Attributes:
        types: boundary condition types for each dimension
        bc_values: boundary values that can be:
            - torch.Tensor: precomputed values along boundary
            - None: homogeneous boundary condition

    Example usage:
        # Array-based boundary conditions
        grid = Grid((10, 20))
        x_boundary = torch.linspace(0, 1, 20)  # values along y-axis
        y_boundary = torch.sin(torch.linspace(0, 2*np.pi, 10))  # values along x-axis

        bc = VariableBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.NEUMANN, BCType.DIRICHLET)),
            values=((y_boundary, y_boundary),  # left/right boundaries
                    (None, x_boundary))        # bottom/top boundaries
        )
    """

    _types: Tuple[Tuple[str, str], ...]
    _bc_values: Tuple[Tuple[Union[float, BCValue], Union[float, BCValue]], ...]
    ndim: int  # default 2d, dataclass init=False

    def __init__(
        self,
        types: Sequence[Tuple[str, str]],
        values: Sequence[
            Tuple[
                Union[float, BCValue],
                Union[float, BCValue],
            ]
        ],
    ):
        types = tuple(types)
        values = tuple(values)
        object.__setattr__(self, "_types", types)
        object.__setattr__(self, "_bc_values", values)
        object.__setattr__(self, "ndim", len(types))

    def __hash__(self):
        """Make DiscreteBoundaryConditions hashable."""
        # Hash the types
        types_hash = hash(self.types)

        # Hash boundary values (handling tensors appropriately)
        values_parts = []
        for dim_values in self._bc_values:
            dim_parts = []
            for value in dim_values:
                if isinstance(value, torch.Tensor):
                    # this may be slow
                    dim_parts.append(tuple(value.flatten().tolist()))
                elif value is None:
                    dim_parts.append(None)
                else:
                    dim_parts.append(value)
            values_parts.append(tuple(dim_parts))
        values_hash = hash(tuple(values_parts))

        return hash((types_hash, values_hash, self.ndim))

    def __eq__(self, other):
        """Define equality for DiscreteBoundaryConditions."""
        if not isinstance(other, DiscreteBoundaryConditions):
            return False

        # Check basic attributes
        if self.types != other.types or self.ndim != other.ndim:
            return False

        # Check boundary values
        for dim in range(self.ndim):
            for side in range(2):
                self_val = self._bc_values[dim][side]
                other_val = other._bc_values[dim][side]

                if self_val is None and other_val is None:
                    continue
                elif self_val is None or other_val is None:
                    return False
                elif isinstance(self_val, torch.Tensor) and isinstance(
                    other_val, torch.Tensor
                ):
                    if not torch.equal(self_val, other_val):
                        return False
                elif self_val != other_val:
                    return False

        return True

    @property
    def has_callable(self) -> bool:
        """Check if any boundary values are callable functions."""
        for dim in range(self.ndim):
            for side in range(2):
                if callable(self._bc_values[dim][side]):
                    return True
        return False

    def _validate_boundary_arrays_with_grid(self, grid: Grid):
        """Validate boundary arrays against grid dimensions."""
        for dim in range(self.ndim):
            for side in range(2):
                value = self._bc_values[dim][side]
                if isinstance(value, torch.Tensor):
                    # Calculate expected boundary shape
                    expected_shape = grid.shape[:dim] + grid.shape[dim + 1 :]
                    if len(expected_shape) == 0:
                        # 1D case - boundary is a scalar
                        if value.numel() != 1:
                            raise ValueError(
                                f"Boundary array for 1D grid at dim {dim}, side {side} "
                                f"should be a scalar, got shape {value.shape}"
                            )
                    elif value.ndim == self.ndim - 1 and value.shape != expected_shape:
                        raise ValueError(
                            f"Boundary array for dim {dim}, side {side} has shape "
                            f"{value.shape}, expected {expected_shape}"
                        )

    @property
    def bc_values(
        self,
    ) -> Sequence[Tuple[Optional[BCValue], Optional[BCValue]]]:
        """Returns boundary values as tensors for each boundary.

        For callable boundary conditions, this will raise an error asking the user
        to use FunctionBoundaryConditions instead.
        For float boundary conditions, returns tensors with the constant value.
        For tensor boundary conditions, returns them as-is.
        For None, returns None.
        """
        if self.has_callable:
            raise ValueError(
                "Callable boundary conditions detected. Please use "
                "FunctionBoundaryConditions class for callable boundary conditions."
            )

        # Process non-callable values
        result = []
        for dim in range(self.ndim):
            dim_values = []
            for side in range(2):
                value = self._bc_values[dim][side]
                if value is None:
                    dim_values.append(None)
                elif isinstance(value, torch.Tensor):
                    dim_values.append(value)
                elif isinstance(value, numbers.Number):
                    # Return scalar tensor for float values
                    dim_values.append(float(value))
                else:
                    raise ValueError(f"Unsupported boundary value type: {type(value)}")
            result.append(tuple(dim_values))
        return tuple(result)

    def __repr__(self) -> str:
        try:
            lines = [f"VariableBoundaryConditions({self.ndim}D):"]

            for dim in range(self.ndim):
                lower_type, upper_type = self.types[dim]
                lower_val, upper_val = self._bc_values[dim]

                # Format values based on type
                def format_value(val):
                    if val is None:
                        return "None"
                    elif isinstance(val, torch.Tensor):
                        return f"Tensor{tuple(val.shape)}"
                    elif callable(val):
                        return f"Callable({val.__name__ if hasattr(val, '__name__') else 'lambda'})"
                    else:
                        return str(val)

                lower_val_str = format_value(lower_val)
                upper_val_str = format_value(upper_val)

                lines.append(
                    f"  dim {dim}: [{lower_type}({lower_val_str}), {upper_type}({upper_val_str})]"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"VariableBoundaryConditions not initialized: {e}"

    def clone(
        self,
        types: Optional[Sequence[Tuple[str, str]]] = None,
        values: Optional[
            Sequence[
                Tuple[
                    BCValue,
                    BCValue,
                ]
            ]
        ] = None,
    ) -> BoundaryConditions:
        """Creates a copy with optionally modified parameters."""
        new_types = types if types is not None else self.types
        new_values = values if values is not None else self._bc_values
        return DiscreteBoundaryConditions(new_types, new_values)

    def _boundary_slices(
        self, offset: Tuple[float, ...]
    ) -> Tuple[Tuple[Optional[slice], Optional[slice]], ...]:
        """Returns slices for boundary values after considering trimming effects.

        When a GridVariable with certain offsets gets trimmed, the boundary coordinates
        need to be sliced accordingly to match the trimmed interior data.
        Currently, this only works for 2D grids (spatially the variable lives on a 2D grid, i.e., good for 2D+time+channel variables).

        Args:
            offset: The offset of the GridVariable
            grid: The grid associated with the GridVariable

        Returns:
            A tuple of (lower_slice, upper_slice) for each dimension, where each slice
            indicates how to index the boundary values for that dimension and side.
            (None, None) means no slicing needed (use full boundary array).
        """
        if self.ndim > 2:
            raise NotImplementedError(
                "Multi-dimensional boundary slicing not implemented"
            )
        if len(offset) != self.ndim:
            raise ValueError(
                f"Offset length {len(offset)} doesn't match number of sets of boundary edges {self.ndim}"
            )

        # Initialize with default "no slicing" tuples
        slices: List[Tuple[Optional[slice], Optional[slice]]] = [
            (None, None),
            (None, None),
        ]

        for dim in range(self.ndim):
            other_dim = dim ^ 1  # flip the bits to get the other dimension index
            trimmed_lower = math.isclose(offset[dim], 0.0)
            trimmed_upper = math.isclose(offset[dim], 1.0)

            assert not (
                trimmed_lower and trimmed_upper
            ), "MAC grids cannot ahve both lower and upper trimmed for bc."
            if trimmed_lower:
                slices[other_dim] = (slice(1, None), slice(1, None))
            elif trimmed_upper:
                slices[other_dim] = (slice(None, -1), slice(None, -1))
            # else: keep the default (None, None)

        return tuple(slices)

    def _boundary_mesh(
        self,
        dim: int,
        grid: Grid,
        offset: Tuple[float, ...],
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Get coordinate arrays for boundary points along dimension dim."""
        # Use the Grid's boundary_mesh method and return coordinates for lower boundary
        # (both lower and upper have same coordinate structure for the boundary points)
        return grid.boundary_mesh(dim, offset)

    def pad_and_impose_bc(
        self,
        u: GridVariable,
        offset_to_pad_to: Optional[Tuple[float, ...]] = None,
        mode: Optional[str] = "",
    ) -> GridVariable:
        """Pad and impose variable boundary conditions."""
        assert u.bc is None, "u must be trimmed before padding and imposing bc."
        if offset_to_pad_to is None:
            offset_to_pad_to = u.offset

        bc_values = self.bc_values
        boundary_slices = self._boundary_slices(offset_to_pad_to)
        x_boundary_slice = boundary_slices[-2]

        if not all(s is None for s in x_boundary_slice):
            # Apply slicing to boundary values
            new_bc_values = list(list(v) for v in bc_values)
            for i in range(2):
                if bc_values[-2][i] is not None:
                    if isinstance(bc_values[-2][i], torch.Tensor):
                        if bc_values[-2][i].ndim > 0:
                            new_bc_values[-2][i] = bc_values[-2][i][x_boundary_slice[i]]
                        elif bc_values[-2][i].ndim == 0:
                            new_bc_values[-2][i] = bc_values[-2][i].item()
                    else:
                        new_bc_values[-2][i] = bc_values[-2][i]
            bc_values = new_bc_values

        for dim in range(-u.grid.ndim, 0):
            _ = self._is_aligned(u, dim)
            if self.types[dim][0] != BCType.PERIODIC:
                # the values passed to grids.pad should consider the offset of the variable
                # if the offset is 1, the the trimmed variable will have the upper edge of that dimension trimmed, one only needs n-1 entries.
                if mode:
                    u = grids.pad(u, (1, 1), dim, self, mode=mode, values=bc_values)
                elif self.types[dim][0] == BCType.DIRICHLET and not mode:
                    if self.types[dim][1] == BCType.DIRICHLET and math.isclose(offset_to_pad_to[dim], 1.0):
                        u = grids.pad(u, 1, dim, self, values=bc_values)
                    elif math.isclose(offset_to_pad_to[dim], 0.0):
                        u = grids.pad(u, -1, dim, self, values=bc_values)

        return GridVariable(u.data, u.offset, u.grid, self)


@dataclasses.dataclass(init=False, frozen=True, repr=False)
class FunctionBoundaryConditions(DiscreteBoundaryConditions):
    """Boundary conditions defined by callable functions.

    This class handles boundary conditions that are defined as functions of
    spatial coordinates (and optionally time). The functions are automatically
    evaluated on the boundary mesh during initialization.

    Attributes:
        types: boundary condition types for each dimension
        _bc_values: evaluated boundary values (tensors/floats after evaluation)
        ndim: number of spatial dimensions

    Example usage:
        # Function-based boundary conditions with individual functions
        def left_bc(x, y):
            return torch.sin(y)

        def right_bc(x, y):
            return torch.cos(y)

        grid = Grid((10, 20))
        bc = FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.NEUMANN, BCType.DIRICHLET)),
            values=((left_bc, right_bc),    # left/right boundaries
                    (None, lambda x, y: x**2))  # bottom/top boundaries
            grid=grid,
            offset=(0.5, 0.5)
        )

        # Or with a single function applied to all boundaries
        def global_bc(x, y):
            return x + y

        bc = FunctionBoundaryConditions(
            types=((BCType.DIRICHLET, BCType.DIRICHLET),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=global_bc,  # Single function applied everywhere
            grid=grid,
            offset=(0.5, 0.5)
        )
    """

    _raw_bc_values: Tuple[
        Tuple[
            Union[
                Callable[..., torch.Tensor],
                Union[Callable[..., torch.Tensor], BCValue, float],
            ],
            BCValue,
            float,
        ],
        ...,
    ]

    def __init__(
        self,
        types: Sequence[Tuple[str, str]],
        values: Union[
            Callable[..., torch.Tensor],  # Single function for all boundaries
            Sequence[
                Tuple[
                    Union[Callable[..., torch.Tensor], BCValue, float],
                    Union[Callable[..., torch.Tensor], BCValue, float],
                ]
            ],
        ],
        grid: Grid,
        offset: Optional[Tuple[float, ...]] = None,
        time: Optional[torch.Tensor] = None,
    ):
        """Initialize function-based boundary conditions.

        Args:
            types: boundary condition types for each dimension
            values: boundary values that can be:
                - Single Callable: function to apply to all boundaries
                - Sequence of tuples: individual values per boundary that can be:
                    - Callable: function to evaluate on boundary mesh
                    - torch.Tensor: precomputed values along boundary
                    - float/int: constant value
                    - None: homogeneous boundary condition
            grid: Grid to evaluate boundary conditions on
            offset: Grid offset for boundary coordinate calculation
            time: Optional time parameter for time-dependent boundary conditions
        """
        types = tuple(types)

        # Handle single callable function case
        if callable(values):
            # Apply the same function to all boundaries
            ndim = len(types)
            values = tuple((values, values) for _ in range(ndim))
        else:
            values = tuple(values)

        # Set basic attributes first
        object.__setattr__(self, "_types", types)
        object.__setattr__(self, "ndim", len(types))
        object.__setattr__(self, "_raw_bc_values", values)

        if offset is None:
            offset = grid.cell_center

        # Evaluate callable boundary conditions
        evaluated_values = []

        for dim in range(len(types)):
            dim_values = []

            # Get boundary coordinates for this dimension if needed
            boundary_coords = None

            for side in range(2):
                value = values[dim][side]

                if value is None:
                    dim_values.append(None)
                elif isinstance(value, torch.Tensor):
                    dim_values.append(value)
                elif isinstance(value, (int, float)):
                    dim_values.append(torch.tensor(float(value)))
                elif isinstance(value, Callable):
                    # Get boundary coordinates if not already computed
                    if boundary_coords is None:
                        lower_coords, upper_coords = grid.boundary_mesh(dim, offset)
                        boundary_coords = (lower_coords, upper_coords)

                    # Evaluate callable on appropriate boundary
                    boundary_points = boundary_coords[side]
                    if time is not None:
                        evaluated_value = value(*boundary_points, t=time)
                    else:
                        evaluated_value = value(*boundary_points)
                    dim_values.append(evaluated_value)
                else:
                    raise ValueError(f"Unsupported boundary value type: {type(value)}")

            evaluated_values.append(tuple(dim_values))

        # Set the evaluated values
        object.__setattr__(self, "_bc_values", tuple(evaluated_values))

        # Validate the evaluated arrays
        self._validate_boundary_arrays_with_grid(grid)

    @property
    def has_callable(self) -> bool:
        """Always returns False since all callables are evaluated during init."""
        return False

    @property
    def bc_values(
        self,
    ) -> Sequence[Tuple[Optional[BCValue], Optional[BCValue]]]:
        """Returns boundary values as tensors for each boundary.

        Since all callable functions are evaluated during initialization,
        this property will never encounter callable values and always returns
        the evaluated tensor/float values.
        """
        # Process all values (no callables should exist at this point)
        result = []
        for dim in range(self.ndim):
            dim_values = []
            for side in range(2):
                value = self._bc_values[dim][side]
                if value is None:
                    dim_values.append(None)
                elif isinstance(value, torch.Tensor):
                    dim_values.append(value)
                elif isinstance(value, (int, float)):
                    # Return scalar tensor for float values
                    dim_values.append(torch.tensor(float(value)))
                else:
                    raise ValueError(
                        f"Unexpected boundary value type after evaluation: {type(value)}"
                    )
            result.append(tuple(dim_values))
        return tuple(result)


def dirichlet_boundary_conditions_nonhomogeneous(
    ndim: int,
    bc_values: Sequence[Tuple[BCValue, BCValue]],
) -> DiscreteBoundaryConditions:
    """Create variable Dirichlet boundary conditions."""
    types = ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim
    return DiscreteBoundaryConditions(types, bc_values)


def neumann_boundary_conditions_nonhomogeneous(
    ndim: int,
    bc_values: Sequence[Tuple[BCValue, BCValue],],
) -> DiscreteBoundaryConditions:
    """Create variable Neumann boundary conditions."""
    types = ((BCType.NEUMANN, BCType.NEUMANN),) * ndim
    return DiscreteBoundaryConditions(types, bc_values)


def function_boundary_conditions_nonhomogeneous(
    ndim: int,
    bc_function: Callable[..., torch.Tensor],
    bc_type: str,
    grid: Grid,
    offset: Optional[Tuple[float, ...]] = None,
    time: Optional[torch.Tensor] = None,
) -> FunctionBoundaryConditions:
    """Create function boundary conditions with the same function applied to all boundaries."""
    types = ((bc_type, bc_type),) * ndim
    return FunctionBoundaryConditions(types, bc_function, grid, offset, time)


@dataclasses.dataclass(init=False, frozen=True, repr=False)
class ImmersedBoundaryConditions(DiscreteBoundaryConditions):
    """Boundary conditions with immersed obstacles in the domain.

    This class combines the functionality of DiscreteBoundaryConditions with
    immersed boundary methods for handling obstacles within the computational domain.
    The implementation is based on:
        - An un-merged PR of Jax-CFD: https://github.com/google/jax-cfd/pull/250
        - This PR's code behavior is largely improved by adopting the logic from Distmesh.

    Attributes:
        types: boundary condition types for each dimension
        _bc_values: boundary values for domain boundaries
        center: center coordinates of the immersed shape
        radius: radius (for circle) or half-width (for square) of the immersed shape
        num_obstacles: number of immersed obstacles (default: 1)
        shape: shape type ('circle' or 'square')
        immersed_bc_values: value enforced inside the immersed solid region
        mask: GridVariable indicating fluid (1.0) and solid (0.0) regions (auto-generated)
        ndim: number of spatial dimensions

    Example usage:
        # Create boundary conditions with immersed circle
        bc = ImmersedBoundaryConditions(
            types=((BCType.PERIODIC, BCType.PERIODIC),
                   (BCType.DIRICHLET, BCType.DIRICHLET)),
            values=((None, None), (0.0, 1.0)),  # periodic x, Dirichlet y
            center=(0.5, 0.5),
            radius=0.1,
            shape='circle',
            immersed_bc_values=0.0  # no-slip inside obstacle
        )
    """
    num_obstacles: int  
    center: Sequence[Tuple[float, ...]]
    radius: Sequence[float]
    immersed_shape: str
    immersed_bc_value: float
    mask: Optional[GridVariable]  # Auto-generated from center, radius, shape

    def __init__(
        self,
        types: Sequence[Tuple[str, str]],
        values: Sequence[
            Tuple[
                Union[float, BCValue],
                Union[float, BCValue],
            ]
        ],
        center: Sequence[Tuple[float, ...]],
        radius: Sequence[float],
        num_obstacles: int = 1,
        shape: str = "circle",
        immersed_bc_value: float = 0.0,
        grid: Optional[Grid] = None,
        offset: Optional[Tuple[float, ...]] = None,
    ):
        """Initialize immersed boundary conditions.

        Args:
            types: boundary condition types for each dimension
            values: boundary values for domain boundaries
            center: center coordinates of the immersed shape
            radius: radius (for circle) or half-width (for square)
            immersed_shape: shape type ('circle' or 'square')
            immersed_bc_values: scalar or tensor value enforced inside solid regions
            grid: computational grid (required for mask generation)
            offset: grid offset for mask evaluation (default: cell center)
        """
        # Initialize parent class
        super().__init__(types, values)

        # Set immersed boundary attributes
        object.__setattr__(self, "num_obstacles", num_obstacles)
        object.__setattr__(self, "center", tuple(center))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "immersed_shape", shape.lower())
        object.__setattr__(self, "immersed_bc_value", immersed_bc_value)

        # Generate mask if grid is provided
        if grid is not None:
            mask = self._create_mask(grid, offset)
            self._validate_mask(mask)
            object.__setattr__(self, "mask", mask)
        else:
            object.__setattr__(self, "mask", None)

    def _create_mask(
        self, grid: Grid, offset: Optional[Tuple[float, ...]] = None
    ) -> GridVariable:
        """Create immersed boundary mask based on center, radius, and shape."""
        
        if self.immersed_shape == "circle":
            dist_func = l2_dist
        elif self.immersed_shape == "square":
            dist_func = linf_dist
        else:
            raise ValueError(
                f"Unsupported shape: {self.immersed_shape}. Use 'circle' or 'square'."
            )
        
        # Start with the first obstacle
        mask = create_immersed_mask(grid, self.center[0], self.radius[0], self, offset, dist_func)
        
        if self.num_obstacles == 1:
            return mask
        elif self.num_obstacles > 1:
            # Process remaining obstacles
            for i in range(1, self.num_obstacles):
                if self.center[i] != self.center[0] or (self.radius[i] > self.radius[0]):
                    mask = create_immersed_mask(grid, self.center[i], self.radius[i], self, offset, dist_func, old_mask=mask)
            # Return the final mask after processing all obstacles
            return mask
        else:
            raise ValueError(f"Number of obstacles has to be >=1, got {self.num_obstacles}.")
        

    def with_grid(
        self, grid: Grid, offset: Optional[Tuple[float, ...]] = None
    ) -> "ImmersedBoundaryConditions":
        """Create a new ImmersedBoundaryConditions with mask generated for the given grid."""
        return ImmersedBoundaryConditions(
            types=self.types,
            values=self._bc_values,
            center=self.center,
            radius=self.radius,
            num_obstacles=self.num_obstacles,
            shape=self.immersed_shape,
            immersed_bc_value=self.immersed_bc_value,
            grid=grid,
            offset=offset,
        )

    def _validate_mask(self, mask: GridVariable):
        """Validate that the mask has appropriate properties."""
        # Check that mask values are in [0, 1]
        if not (torch.all(mask.data >= 0.0) and torch.all(mask.data <= 1.0)):
            raise ValueError("Mask values must be in range [0.0, 1.0]")

    def __hash__(self):
        """Make ImmersedBoundaryConditions hashable by excluding the mask.
        Without implementating this, mask will cause the class un-hashable
        and raises ValueError in {u.bc for u in v} that is used in checking 
        bc consistency.
        """
        # Use parent class hash (which handles types and values)
        parent_hash = super().__hash__()

        # Hash immersed-specific attributes (excluding mask)
        center_hash = hash(tuple(self.center))
        radius_hash = hash(tuple(self.radius))
        shape_hash = hash(self.immersed_shape)
        immersed_hash = hash(self.immersed_bc_value)

        return hash((parent_hash, center_hash, radius_hash, shape_hash, immersed_hash))

    def __eq__(self, other):
        if not isinstance(other, ImmersedBoundaryConditions):
            return False

        # Check parent equality
        if not super().__eq__(other):
            return False

        # Check immersed-specific attributes (excluding mask)
        if (
            self.center != other.center
            or self.radius != other.radius
            or self.immersed_shape != other.immersed_shape
        ):
            return False

        # Check immersed BC values
        return self.immersed_bc_value == other.immersed_bc_value

    def __repr__(self) -> str:
        try:
            lines = [f"ImmersedBoundaryConditions({self.ndim}D):"]

            # Domain boundary conditions
            lines.append("Domain boundaries:")
            for dim in range(self.ndim):
                lower_type, upper_type = self.types[dim]
                lower_val, upper_val = self._bc_values[dim]

                def format_value(val):
                    if val is None:
                        return "None"
                    elif isinstance(val, torch.Tensor):
                        return f"Tensor{tuple(val.shape)}"
                    elif callable(val):
                        return f"Callable({val.__name__ if hasattr(val, '__name__') else 'lambda'})"
                    else:
                        return str(val)

                lower_val_str = format_value(lower_val)
                upper_val_str = format_value(upper_val)

                lines.append(
                    f"  dim {dim}: [{lower_type}({lower_val_str}), {upper_type}({upper_val_str})]"
                )

            # Immersed boundary info
            lines.append(f"Immersed boundary:")
            lines.append(f"  shape: {self.immersed_shape}")
            lines.append(f"  center: {self.center}")
            lines.append(f"  radius: {self.radius}")
            lines.append(f"  immersed BC value: {self.immersed_bc_value}")

            if self.mask is not None:
                solid_fraction = (1.0 - self.mask.data).mean().item()
                lines.append(f"  mask: present (solid fraction: {solid_fraction:.3f})")
                lines.append(f"  mask shape: {tuple(self.mask.shape)}")
            else:
                lines.append(f"  mask: None (call with_grid() to generate)")

            return "\n".join(lines)
        except Exception as e:
            return f"ImmersedBoundaryConditions not initialized: {e}"

    def clone(
        self,
        types: Optional[Sequence[Tuple[str, str]]] = None,
        values: Optional[
            Sequence[
                Tuple[
                    BCValue,
                    BCValue,
                ]
            ]
        ] = None,
        center: Optional[Sequence[Tuple[float, ...]]] = None,
        radius: Optional[Sequence[float]] = None,
        num_obstacles: Optional[int] = 1,
        shape: Optional[str] = None,
        immersed_bc_values: Optional[float] = 0.0,
        grid: Optional[Grid] = None,
        offset: Optional[Tuple[float, ...]] = None,
    ) -> BoundaryConditions:
        """Creates a copy with optionally modified parameters."""
        new_types = types if types is not None else self.types
        new_values = values if values is not None else self._bc_values
        new_center = center if center is not None else self.center
        new_radius = radius if radius is not None else self.radius
        new_num_obstacles = num_obstacles if num_obstacles is not None else self.num_obstacles
        new_shape = shape if shape is not None else self.immersed_shape
        new_immersed_bc_values = (
            immersed_bc_values
            if immersed_bc_values is not None
            else self.immersed_bc_value
        )

        # Use existing grid from mask if grid not provided and mask exists
        if grid is None and self.mask is not None:
            grid = self.mask.grid

        return ImmersedBoundaryConditions(
            new_types,
            new_values,
            new_center,
            new_radius,
            new_num_obstacles,
            new_shape,
            new_immersed_bc_values,
            grid,
            offset,
        )

    def pad_and_impose_bc(
        self,
        u: GridVariable,
        offset_to_pad_to: Optional[Tuple[float, ...]] = None,
        mode: Optional[str] = "",
    ) -> GridVariable:
        """Pad and impose boundary conditions including immersed boundaries."""
        # First apply domain boundary conditions using parent method
        u_with_bc = super().pad_and_impose_bc(u, offset_to_pad_to, mode)

        # Then apply immersed boundary conditions if mask is present
        if self.mask is not None:
            u_with_bc = self.impose_immersed_bc(u_with_bc)
        return u_with_bc

    def impose_immersed_bc(self, u: GridVariable) -> GridVariable:
        """Apply immersed boundary conditions to data tensor."""
        if self.mask is None:
            return u

        data = u.data
        # Handle different shapes between data and mask
        mask = self.mask.data

        # If data has batch dimensions, expand mask accordingly
        if data.ndim > mask.ndim:
            # Assume batch dimensions are at the beginning
            batch_dims = data.shape[: -mask.ndim]
            expanded_shape = batch_dims + mask.shape
            mask = mask.expand(expanded_shape)

        # Apply immersed boundary condition
        # mask_data: 1.0 = fluid, 0.0 = solid
        # In solid regions, set values to immersed_bc_value
        bc_value = self.immersed_bc_value
        data = mask * data + (1.0 - mask) * bc_value
        return GridVariable(data, u.offset, u.grid, self)


def l2_dist(*coords, center):
    squared_dists = [(coord - center[i]) ** 2 for i, coord in enumerate(coords)]
    return torch.sqrt(torch.stack(squared_dists).sum(dim=0))

def linf_dist(*coords, center):
    abs_dists = [torch.abs(coord - center[i]) for i, coord in enumerate(coords)]
    return torch.max(torch.stack(abs_dists, dim=0), dim=0)[0]

def create_immersed_mask(
    grid: Grid,
    center: Tuple[float, ...],
    radius: float,
    bc: BoundaryConditions,
    offset: Optional[Tuple[float, ...]] = None,
    dist_func: Callable[..., torch.Tensor] = l2_dist,
    old_mask: Optional[GridVariable] = None,
) -> GridVariable:
    """Create an immersed boundary mask using a distance function approach.

    Args:
        grid: computational grid
        center: center coordinates of the shape
        radius: radius (for circle) or half-width (for square) of the shape
        bc: boundary conditions to determine mask shape
        offset: grid offset for mask evaluation (default: cell center)
        dist_func: distance function that takes coordinate tensors and returns
                  distance from center.
                  
    Returns:
        GridVariable with 1.0 for fluid regions, 0.0 for solid regions
        
    Notes:
        By default, the mask is cell-centered with shape (nx, ny).
        Special cases where mask has shape (nx+1, ny) or (nx, ny+1):
        - Lower Dirichlet + Upper Neumann with offset 0
        - Lower Neumann + Upper Dirichlet with offset 1
    """
    if offset is None:
        offset = grid.cell_center

    diam = min(grid.domain[0][1] - grid.domain[0][0], 
               grid.domain[1][1] - grid.domain[1][0])
    if radius > 0.25 * diam:
        raise ValueError(
            f"Radius {radius} is too large for the grid domain {grid.domain}. "
            f"Maximum allowed radius is 0.25 * min(domain lengths) = {0.25 * diam}."
        )

    # Start with standard mesh coordinates
    mesh_coords = list(grid.mesh(offset))

    # Handle each dimension separately
    for dim in range(grid.ndim):
        lower_bc, upper_bc = bc.types[dim]
        
        # Case 1: Lower Dirichlet + Upper Neumann with offset 0 → extend left
        if (lower_bc == BCType.DIRICHLET and 
            upper_bc == BCType.NEUMANN and 
            math.isclose(offset[dim], 0.0)):
            
            # Get the leftmost coordinate value
            leftmost = grid.domain[dim][0]
            # Create extension tensor with same shape as mesh_coords[dim] but only first slice
            extension_shape = list(mesh_coords[dim].shape)
            extension_shape[dim] = 1
            extension = torch.full(extension_shape, leftmost, device=grid.device)
            # Concatenate to the left
            mesh_coords[dim] = torch.cat([extension, mesh_coords[dim]], dim=dim)
            
        # Case 2: Lower Neumann + Upper Dirichlet with offset 1 → extend right
        elif (lower_bc == BCType.NEUMANN and 
              upper_bc == BCType.DIRICHLET and 
              math.isclose(offset[dim], 1.0)):
            
            # Get the rightmost coordinate value
            rightmost = grid.domain[dim][1]
            # Create extension tensor with same shape as mesh_coords[dim] but only last slice
            extension_shape = list(mesh_coords[dim].shape)
            extension_shape[dim] = 1
            extension = torch.full(extension_shape, rightmost, device=grid.device)
            # Concatenate to the right
            mesh_coords[dim] = torch.cat([mesh_coords[dim], extension], dim=dim)

    # Calculate mask using distance function
    distance = dist_func(*mesh_coords, center=center)
    mask = torch.where(distance <= radius, 0.0, 1.0)

    if old_mask is None:
        return GridVariable(mask, offset, grid)
    else:
        # If an old mask is provided, use it to determine the solid region
        old_mask_data = old_mask.data
        if old_mask_data.shape != mask.shape:
            raise ValueError(
                f"Existing mask shape {old_mask_data.shape} does not match new mask shape {mask.shape}"
            )
        mask = torch.where(old_mask_data == 0.0, 0.0, mask)
        return GridVariable(mask, offset, grid)


def channel_flow_2d_immersed_body_boundary_conditions(
    grid: Grid,
    bc_vals: Optional[Sequence[Tuple[BCValue, BCValue]]] = None,
    shape: str = "circle",
    radius: float = 0.25,
    immersed_bc_value: float = 0.0,
    center: Optional[Tuple[float, ...]] = None,
    offset: Optional[Tuple[float, ...]] = None,
) -> ImmersedBoundaryConditions:
    """Create channel flow boundary conditions with an immersed obstacle."""
    # Set default center if not provided
    if center is None:
        domain_lengths = [
            grid.domain[i][1] - grid.domain[i][0] for i in range(grid.ndim)
        ]
        center = tuple(
            grid.domain[i][0] + 0.5 * domain_lengths[i] for i in range(grid.ndim)
        )

    # Set up channel flow boundary types (periodic x, Dirichlet y)
    types = ((BCType.PERIODIC, BCType.PERIODIC), (BCType.DIRICHLET, BCType.DIRICHLET))

    # Set default values if not provided
    if bc_vals is None:
        values = ((None, None), (0.0, 0.0))
    else:
        values = bc_vals

    return ImmersedBoundaryConditions(
        types=types,
        values=values,
        center=(center, ),
        radius=(radius, ),
        shape=shape,
        immersed_bc_value=immersed_bc_value,
        grid=grid,
        offset=offset,
    )


def karman_vortex_velocity_boundary_conditions(
    grid: Grid,
    inlet_velocity: Tuple[float, float] = (1.0, 0.0),
    cylinder_center: Tuple[float, float] = (0.4, 0.5),
    cylinder_radius: float = 0.05,
    immersed_bc_values: float = 0.0,
) -> Tuple[ImmersedBoundaryConditions, ImmersedBoundaryConditions]:
    """Create separate velocity boundary conditions for u and v components in 2d von Karman vortex street simulation."""
    if grid.ndim != 2:
        raise ValueError(
            "Karman vortex boundary conditions are only valid for 2D grids"
        )

    # U-velocity boundary conditions
    u_types = (
        (
            BCType.DIRICHLET,
            BCType.NEUMANN,
        ),  # x-direction: inlet Dirichlet, outlet Neumann
        (
            BCType.NEUMANN,
            BCType.NEUMANN,
        ),  # y-direction: slip walls (zero normal gradient)
    )
    u_values = (
        (inlet_velocity[0], 0.0),  # x-direction: inlet u-velocity, zero gradient outlet
        (0.0, 0.0),  # y-direction: zero gradient at walls
    )

    # V-velocity boundary conditions
    v_types = (
        (
            BCType.DIRICHLET,
            BCType.NEUMANN,
        ),  # x-direction: inlet Dirichlet, outlet Neumann
        (BCType.DIRICHLET, BCType.DIRICHLET),  # y-direction: no penetration at walls
    )
    v_values = (
        (inlet_velocity[1], 0.0),  # x-direction: inlet v-velocity, zero gradient outlet
        (0.0, 0.0),  # y-direction: zero normal velocity at walls
    )

    u_bc = ImmersedBoundaryConditions(
        types=u_types,
        values=u_values,
        center=(cylinder_center, ),
        radius=(cylinder_radius, ),
        num_obstacles=1,
        shape="circle",
        immersed_bc_value=immersed_bc_values,
        grid=grid,
        offset=(1, 0.5),
    )

    v_bc = ImmersedBoundaryConditions(
        types=v_types,
        values=v_values,
        center=(cylinder_center, ),
        radius=(cylinder_radius, ),
        num_obstacles=1,
        shape="circle",
        immersed_bc_value=immersed_bc_values,
        grid=grid,
        offset=(0.5, 1),
    )

    return u_bc, v_bc

def karman_vortex_boundary_conditions(
    grid: Grid,
    inlet_velocity: Tuple[float, float] = (1.0, 0.0),
    inlet_pressure: float = 0.0,
    outlet_pressure: float = 0.0,
    cylinder_center: Tuple[float, float] = (0.4, 0.5),
    cylinder_radius: float = 0.05,
    immersed_bc_value: float = 0.0,
) -> Tuple[Tuple[ImmersedBoundaryConditions, ...], ConstantBoundaryConditions]:
    """Create complete set of boundary conditions for Kármán vortex street.

    Returns:
        Tuple of (velocity_bc, pressure_bc) boundary conditions
    """
    u_bc, v_bc = karman_vortex_velocity_boundary_conditions(
        grid,
        inlet_velocity,
        cylinder_center,
        cylinder_radius,
        immersed_bc_value,
    )
    p_bc = ConstantBoundaryConditions(
        types=((BCType.NEUMANN, BCType.DIRICHLET,),
        (BCType.NEUMANN, BCType.NEUMANN)),
        values=((inlet_pressure, outlet_pressure),
        (0.0, 0.0)),
    )
    return (u_bc, v_bc), p_bc


def cavity_flow_2d_boundary_conditions(
    grid: Grid,
    lid_velocity: Tuple[float, float] = (1.0, 0.0),
    wall_velocity: Tuple[float, float] = (0.0, 0.0),
    smooth: bool = False,
) -> Tuple[BoundaryConditions, BoundaryConditions]:
    """Create boundary conditions for 2D lid-driven cavity flow.
    
    Creates Dirichlet boundary conditions for all walls with specified velocities.
    The top wall (lid) moves with lid_velocity, while other walls are stationary
    with wall_velocity (typically zero).
    
    Args:
        grid: computational grid
        lid_velocity: velocity of the top wall (lid), default (1.0, 0.0)
        wall_velocity: velocity of other walls, default (0.0, 0.0)
        smooth: if True, applies parabolic profile 4*x*(1-x) to lid velocity
        
    Returns:
        Tuple of (u_bc, v_bc) boundary conditions:
        - ConstantBoundaryConditions for regular cavity flow (smooth=False)
        - FunctionBoundaryConditions for smooth lid (smooth=True)
        
    Example:
        # Regular lid-driven cavity
        u_bc, v_bc = cavity_flow_2d_boundary_conditions(grid)
        
        # Smooth lid-driven cavity
        u_bc, v_bc = cavity_flow_2d_boundary_conditions(grid, smooth=True)
    """
    if grid.ndim != 2:
        raise ValueError("Cavity flow boundary conditions are only valid for 2D grids")
    
    # Set up cavity flow boundary types (all Dirichlet)
    types = (
        (BCType.DIRICHLET, BCType.DIRICHLET),  # x-direction: left/right walls
        (BCType.DIRICHLET, BCType.DIRICHLET),  # y-direction: bottom/top walls
    )
    
    if smooth:
        # Define smooth lid velocity functions

        top_u_velocity = lambda x, y: 4 * x * (1 - x) * lid_velocity[0]
        zero_velocity = lambda x, y: torch.zeros_like(x)
        
        # For u-velocity component with smooth lid
        u_values = (
            (zero_velocity, zero_velocity),  # left/right walls: zero u-velocity
            (zero_velocity, top_u_velocity),  # bottom/top walls: zero bottom, smooth top
        )
        
        # For v-velocity component (always zero for cavity flow)
        v_values = (
            (zero_velocity, zero_velocity),  # left/right walls: zero v-velocity
            (zero_velocity, zero_velocity),  # bottom/top walls: zero v-velocity
        )
        
        u_offset = (1.0, 0.5)  # u-velocity staggered grid offset
        v_offset = (0.5, 1.0)  # v-velocity staggered grid offset
        
        u_bc = FunctionBoundaryConditions(types, u_values, grid, u_offset)
        v_bc = FunctionBoundaryConditions(types, v_values, grid, v_offset)
        return (u_bc, v_bc)
    else:
        # Constant boundary values
        # For u-velocity component
        u_values = (
            (wall_velocity[0], wall_velocity[0]),  # u-velocity on left/right walls
            (wall_velocity[0], lid_velocity[0]),   # u-velocity on bottom/top walls (lid moves)
        )
        
        # For v-velocity component  
        v_values = (
            (wall_velocity[1], wall_velocity[1]),  # v-velocity on left/right walls
            (wall_velocity[1], lid_velocity[1]),   # v-velocity on bottom/top walls
        )
        
        u_bc = ConstantBoundaryConditions(types, u_values)
        v_bc = ConstantBoundaryConditions(types, v_values)
        return (u_bc, v_bc)


def cavity_flow_2d_boundary_conditions_with_obstacles(
    grid: Grid,
    obstacle_centers: Sequence[Tuple[float, float]],
    obstacle_radius: Union[float, Sequence[float]],
    lid_velocity: Tuple[float, float] = (1.0, 0.0),
    wall_velocity: Tuple[float, float] = (0.0, 0.0),
    smooth: bool = False,
    obstacle_shape: str = "square",
    immersed_bc_value: float = 0.0,
) -> Tuple[ImmersedBoundaryConditions, ImmersedBoundaryConditions]:
    """Create boundary conditions for 2D lid-driven cavity flow with immersed obstacles.
    
    Creates Dirichlet boundary conditions for all walls with specified velocities
    and immersed obstacles within the domain. The top wall (lid) moves with 
    lid_velocity, while other walls are stationary with wall_velocity.
    
    Args:
        grid: computational grid
        obstacle_centers: center coordinates of obstacles
        obstacle_radius: radius (circle) or half-width (square) of obstacles.
                        Can be single value or sequence for multiple obstacles
        lid_velocity: velocity of the top wall (lid), default (1.0, 0.0)
        wall_velocity: velocity of other walls, default (0.0, 0.0)
        smooth: if True, applies parabolic profile 4*x*(1-x) to lid velocity
        obstacle_shape: shape of obstacles ('circle' or 'square'), default 'square'
        immersed_bc_value: velocity value inside solid regions, default 0.0
        
    Returns:
        Tuple of (u_bc, v_bc) ImmersedBoundaryConditions
        
    Example:
        # Cavity with one square obstacle
        u_bc, v_bc = cavity_flow_2d_boundary_conditions_with_obstacles(
            grid, 
            obstacle_centers=[(0.5, 0.5)],
            obstacle_radius=0.15
        )
        
        # Smooth cavity with two circular obstacles
        u_bc, v_bc = cavity_flow_2d_boundary_conditions_with_obstacles(
            grid, 
            obstacle_centers=[(0.3, 0.3), (0.7, 0.7)],
            obstacle_radius=[0.1, 0.12],
            smooth=True,
            obstacle_shape="circle"
        )
    """
    if grid.ndim != 2:
        raise ValueError("Cavity flow boundary conditions are only valid for 2D grids")
    
    num_obstacles = len(obstacle_centers)
    if num_obstacles == 0:
        raise ValueError("Use cavity_flow_2d_boundary_conditions() for cavity flow without obstacles")
    
    # Set up cavity flow boundary types (all Dirichlet)
    types = (
        (BCType.DIRICHLET, BCType.DIRICHLET),  # x-direction: left/right walls
        (BCType.DIRICHLET, BCType.DIRICHLET),  # y-direction: bottom/top walls
    )
    
    # Handle obstacle radius
    if isinstance(obstacle_radius, (int, float)):
        obstacle_radii = [float(obstacle_radius)] * num_obstacles
    else:
        obstacle_radii = list(obstacle_radius)
        if len(obstacle_radii) != num_obstacles:
            raise ValueError(f"Number of radii ({len(obstacle_radii)}) must match number of obstacles ({num_obstacles})")
    
    # Handle smooth vs non-smooth boundary values
    if smooth:
        # Use grid.axes() to get the x-coordinates for smooth lid profile
        x_coords = grid.axes(offset=(1.0, 0.5))[0]  # u-velocity staggered grid
        
        # Evaluate top boundary function: 4*x*(1-x)*lid_velocity[0]
        top_u_vals = 4 * x_coords * (1 - x_coords) * lid_velocity[0]
        
        # Create boundary values with smooth lid
        u_boundary_values = (
            (wall_velocity[0], wall_velocity[0]),  # left/right walls
            (wall_velocity[0], top_u_vals)         # bottom/top walls (smooth lid)
        )
        
        v_boundary_values = (
            (wall_velocity[1], wall_velocity[1]),  # left/right walls
            (wall_velocity[1], wall_velocity[1])   # bottom/top walls (zero v-velocity)
        )
    else:
        # Constant boundary values
        u_boundary_values = (
            (wall_velocity[0], wall_velocity[0]),  # u-velocity on left/right walls
            (wall_velocity[0], lid_velocity[0]),   # u-velocity on bottom/top walls (lid moves)
        )
        
        v_boundary_values = (
            (wall_velocity[1], wall_velocity[1]),  # v-velocity on left/right walls
            (wall_velocity[1], lid_velocity[1]),   # v-velocity on bottom/top walls
        )
    
    # Create immersed boundary conditions
    u_bc = ImmersedBoundaryConditions(
        types=types,
        values=u_boundary_values,
        center=obstacle_centers,
        radius=obstacle_radii,
        num_obstacles=num_obstacles,
        shape=obstacle_shape.lower(),
        immersed_bc_value=immersed_bc_value,
        grid=grid,
        offset=(1.0, 0.5),  # u-velocity staggered grid offset
    )
    
    v_bc = ImmersedBoundaryConditions(
        types=types,
        values=v_boundary_values,
        center=obstacle_centers,
        radius=obstacle_radii,
        num_obstacles=num_obstacles,
        shape=obstacle_shape.lower(),
        immersed_bc_value=immersed_bc_value,
        grid=grid,
        offset=(0.5, 1.0),  # v-velocity staggered grid offset
    )
    
    return (u_bc, v_bc)

def _count_bc_components(bc: BoundaryConditions) -> int:
    """Counts the number of components in the boundary conditions.

    Returns:
        The number of components in the boundary conditions.
    """
    count = 0
    ndim = len(bc.types)
    for dim in range(ndim):  # ndim
        if len(bc.types[dim]) != 2:
            raise ValueError(
                f"Boundary conditions for axis {dim} must have two values got {len(bc.types[dim])}."
            )
        count += len(bc.types[dim])
    return count


def consistent_boundary_conditions_grid(
    grid, *arrays: GridVariable
) -> Tuple[GridVariable, ...]:
    """Returns the updated boundary condition if the number of components is inconsistent
    with the grid
    """
    bc_counts = []
    for array in arrays:
        bc_counts.append(_count_bc_components(array.bc))
    bc_count = bc_counts[0]
    if any(bc_counts[i] != bc_count for i in range(1, len(bc_counts))):
        raise Exception("Boundary condition counts are inconsistent")
    if any(bc_counts[i] != 2 * grid.ndim for i in range(len(bc_counts))):
        raise ValueError(
            f"Boundary condition counts {bc_counts} are inconsistent with grid dimensions {grid.ndim}"
        )
    return arrays


def consistent_boundary_conditions_gridvariable(
    *arrays: GridVariable,
) -> Tuple[str, ...]:
    """Returns whether BCs are periodic.

    Mixed periodic/nonperiodic boundaries along the same boundary do not make
    sense. The function checks that the boundary is either periodic or not and
    throws an error if its mixed.

    Args:
      *arrays: a list of gridvariables.

    Returns:
      a list of types of boundaries corresponding to each axis if
      they are consistent.
    """
    bc_types = []
    for dim in range(arrays[0].grid.ndim):
        bcs = {is_periodic_boundary_conditions(array, dim) for array in arrays}
        if len(bcs) != 1:
            raise Exception(f"arrays do not have consistent bc: {arrays}")
        elif bcs.pop():
            bc_types.append("periodic")
        else:
            bc_types.append("nonperiodic")
    return tuple(bc_types)


def get_pressure_bc_from_velocity_bc(
    bcs: Tuple[BoundaryConditions, ...],
) -> HomogeneousBoundaryConditions:
    """Returns pressure boundary conditions for the specified velocity BCs.
    if the velocity BC is periodic, the pressure BC is periodic.
    if the velocity BC is nonperiodic, the pressure BC is zero flux (homogeneous Neumann).
    """
    # assumes that if the boundary is not periodic, pressure BC is zero flux.
    pressure_bc_types = []

    for velocity_bc in bcs:
        if is_bc_periodic_boundary_conditions(velocity_bc, 0):
            pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
        else:
            pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))

    return HomogeneousBoundaryConditions(pressure_bc_types)


def get_pressure_bc_from_velocity(
    v: GridVariableVector,
) -> HomogeneousBoundaryConditions:
    """Returns pressure boundary conditions for the specified velocity."""
    # assumes that if the boundary is not periodic, pressure BC is zero flux.
    velocity_bc_types = consistent_boundary_conditions_gridvariable(*v)
    pressure_bc_types = []
    for velocity_bc_type in velocity_bc_types:
        if velocity_bc_type == "periodic":
            pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
        else:
            pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
    return HomogeneousBoundaryConditions(pressure_bc_types)


def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
    """Returns True if arrays have periodic BC in every dimension, else False."""
    for array in arrays:
        for dim in range(array.grid.ndim):
            if not is_periodic_boundary_conditions(array, dim):
                return False
    return True


def get_advection_flux_bc_from_velocity_and_scalar_bc(
    u_bc: BoundaryConditions, c_bc: BoundaryConditions, flux_direction: int
) -> ConstantBoundaryConditions:
    """Returns advection flux boundary conditions for the specified velocity.

    Infers advection flux boundary condition in flux direction
    from scalar c and velocity u in direction flux_direction.
    The flux boundary condition should be used only to compute divergence.
    If the boundaries are periodic, flux is periodic.
    In nonperiodic case, flux boundary parallel to flux direction is
    homogeneous dirichlet.
    In nonperiodic case if flux direction is normal to the wall, the
    function supports multiple cases:
      1) Nonporous boundary, corresponding to homogeneous flux bc.
      2) Porous boundary with constant flux, corresponding to
        both the velocity and scalar with Homogeneous Neumann bc.
      3) Non-homogeneous Dirichlet velocity with Dirichlet scalar bc.

    Args:
      u_bc: bc of the velocity component in flux_direction.
      c_bc: bc of the scalar to advect.
      flux_direction: direction of velocity.

    Returns:
      BoundaryCondition instance for advection flux of c in flux_direction.

    Example:
    >>> u_bc = ConstantBoundaryConditions(((BCType.DIRICHLET, BCType.DIRICHLET),),  ((1.0, 2.0),))

    >>> c_bc = ConstantBoundaryConditions(((BCType.DIRICHLET, BCType.DIRICHLET),), ((0.5, 1.5),))

    >>> flux_bc = get_advection_flux_bc_from_velocity_and_scalar_bc(u_bc, c_bc,0)
    # flux_bc will have values (0.5, 3.0) = (1.0*0.5, 2.0*1.5)
    """
    # Handle immersed boundary conditions by temporarily ignoring the mask
    # for domain flux boundary computation
    u_bc_for_flux = u_bc
    c_bc_for_flux = c_bc

    if isinstance(u_bc, ImmersedBoundaryConditions):
        # Create a temporary BC with the same domain boundary settings but no mask
        u_bc_for_flux = DiscreteBoundaryConditions(
            types=u_bc.types,
            values=u_bc._bc_values,
        )

    if isinstance(c_bc, ImmersedBoundaryConditions):
        # Create a temporary BC with the same domain boundary settings but no mask
        c_bc_for_flux = DiscreteBoundaryConditions(
            types=c_bc.types,
            values=c_bc._bc_values,
        )

    # Now proceed with the usual flux boundary condition logic
    flux_bc_types = []
    flux_bc_values = []

    # Handle both homogeneous and non-homogeneous boundary conditions
    # cannot handle mixed boundary conditions yet.
    if isinstance(u_bc_for_flux, HomogeneousBoundaryConditions):
        u_values = tuple((0.0, 0.0) for _ in range(u_bc_for_flux.ndim))
    elif isinstance(
        u_bc_for_flux, (ConstantBoundaryConditions, DiscreteBoundaryConditions)
    ):
        if hasattr(u_bc_for_flux, "bc_values"):
            u_values = u_bc_for_flux.bc_values
        else:
            u_values = u_bc_for_flux._bc_values
    else:
        raise NotImplementedError(
            f"Flux boundary condition is not implemented for velocity with {type(u_bc_for_flux)}"
        )

    if isinstance(c_bc_for_flux, HomogeneousBoundaryConditions):
        c_values = tuple((0.0, 0.0) for _ in range(c_bc_for_flux.ndim))
    elif isinstance(
        c_bc_for_flux, (ConstantBoundaryConditions, DiscreteBoundaryConditions)
    ):
        if hasattr(c_bc_for_flux, "bc_values"):
            c_values = c_bc_for_flux.bc_values
        else:
            c_values = c_bc_for_flux._bc_values
    else:
        raise NotImplementedError(
            f"Flux boundary condition is not implemented for scalar with {type(c_bc_for_flux)}"
        )

    for dim in range(c_bc_for_flux.ndim):
        if u_bc_for_flux.types[dim][0] == BCType.PERIODIC:
            flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
            flux_bc_values.append((None, None))
        elif flux_direction != dim:
            # Flux boundary condition parallel to flux direction
            # Set to homogeneous Dirichlet as it doesn't affect divergence computation
            flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
            flux_bc_values.append((0.0, 0.0))
        else:
            # Flux direction is normal to the boundary
            flux_bc_types_ax = []
            flux_bc_values_ax = []

            for i in range(2):  # lower and upper boundary
                u_type = u_bc_for_flux.types[dim][i]
                c_type = c_bc_for_flux.types[dim][i]

                # Extract values, handling both tensor and scalar cases
                u_val = u_values[dim][i] if u_values[dim][i] is not None else 0.0
                c_val = c_values[dim][i] if c_values[dim][i] is not None else 0.0

                # Case 1: Dirichlet velocity with Dirichlet scalar
                if u_type == BCType.DIRICHLET and c_type == BCType.DIRICHLET:
                    # Flux = u * c at the boundary
                    flux_val = u_val * c_val
                    flux_bc_types_ax.append(BCType.DIRICHLET)
                    flux_bc_values_ax.append(flux_val)

                # Case 2: Neumann velocity with Neumann scalar (zero flux condition)
                elif u_type == BCType.NEUMANN and c_type == BCType.NEUMANN:
                    # For zero flux: du/dn = 0 and dc/dn = 0 implies d(uc)/dn = 0
                    if not math.isclose(u_val, 0.0) or not math.isclose(c_val, 0.0):
                        raise NotImplementedError(
                            "Non-homogeneous Neumann boundary conditions for flux "
                            "are not yet implemented"
                        )
                    flux_bc_types_ax.append(BCType.NEUMANN)
                    flux_bc_values_ax.append(0.0)

                # Case 3: Mixed boundary conditions
                elif u_type == BCType.DIRICHLET and c_type == BCType.NEUMANN:
                    # If u is specified and dc/dn is not zero, raises NotImplementedError
                    if not math.isclose(u_val, 0.0) or not math.isclose(c_val, 0.0):
                        raise NotImplementedError(
                            "Non-homogeneous mixed Dirichlet velocity and Neumann scalar boundary "
                            "conditions are not yet implemented"
                        )
                    flux_bc_types_ax.append(BCType.DIRICHLET)
                    flux_bc_values_ax.append(0.0)

                elif u_type == BCType.NEUMANN and c_type == BCType.DIRICHLET:
                    # If du/dn is specified and c is specified
                    # This also requires more complex handling
                    raise NotImplementedError(
                        "Mixed Neumann velocity and Dirichlet scalar boundary "
                        "conditions are not yet implemented"
                    )

                else:
                    raise NotImplementedError(
                        f"Flux boundary condition is not implemented for "
                        f"u_bc={u_type} with value={u_val}, "
                        f"c_bc={c_type} with value={c_val}"
                    )

            flux_bc_types.append(tuple(flux_bc_types_ax))
            flux_bc_values.append(tuple(flux_bc_values_ax))
    if any([isinstance(v, torch.Tensor) for v in flux_bc_values]):
        return DiscreteBoundaryConditions(flux_bc_types, flux_bc_values)
    return ConstantBoundaryConditions(flux_bc_types, flux_bc_values)


def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable, flux_direction: int
) -> ConstantBoundaryConditions:
    return get_advection_flux_bc_from_velocity_and_scalar_bc(u.bc, c.bc, flux_direction)
