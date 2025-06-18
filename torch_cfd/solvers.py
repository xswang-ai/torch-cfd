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

"""Collections of linear system solvers."""

from functools import partial, reduce
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.fft as fft
import torch.nn as nn

from torch_cfd import boundaries, finite_differences as fdm, grids

Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


def _set_laplacian(
    module: nn.Module,
    laplacians: List[torch.Tensor] | None,
    grid: Grid,
    bc: BoundaryConditions,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Initialize the 1D Laplacian operators with ndim
    Args:
        laplacians have the shape (ndim, n, n)
    """
    if laplacians is None:
        laplacians = fdm.set_laplacian_matrix(grid, bc, device, dtype)
    else:
        # Check if the provided laplacians are consistent with the grid
        for laplacian in laplacians:
            if laplacian.shape != grid.shape:
                raise ValueError("Provided laplacians do not match the grid shape.")

    # Register each laplacian separately since they may have different sizes
    for i, laplacian in enumerate(laplacians):
        module.register_buffer(f"laplacian_{i}", laplacian, persistent=True)


def outer_sum(x: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Returns the outer sum of a list of one dimensional arrays
    Example:
    x = [a, b, c]
    out = a[..., None, None] + b[..., None] + c

    The full outer sum is equivalent to:
    def _sum(a, b):
        return a[..., None] + b
    return reduce(_sum, x)
    """
    return reduce(lambda a, b: a[..., None] + b, x)


class Identity(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return x


class SolverBase(nn.Module):
    """
    Base class for solvers. This class defines the interface for solvers that apply a linear operator equation on a 2D grid.

    Args:
        grid: Grid object describing the spatial domain.
        bc: Boundary conditions for the Laplacian operator (for pressure).
        dtype: Tensor data type.
        laplacians: Precomputed Laplacian operators. If None, they are computed from
            the grid during initialization.
        tol: Tolerance for filtering eigenvalues in the pseudoinverse/iterative solver's rel residual.
    """

    def __init__(
        self,
        grid: grids.Grid,
        bc: BoundaryConditions | None = None,
        dtype: torch.dtype = torch.float32,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        self.grid = grid
        if bc is None:
            bc = boundaries.periodic_boundary_conditions(ndim=grid.ndim)
        self.bc = bc
        self.ndim = grid.ndim
        self.tol = tol
        self.dtype = dtype
        _set_laplacian(self, laplacians, grid, bc, dtype=dtype)
        self._compute_inverse_diagonals()

    @property
    def operators(self) -> List[torch.Tensor]:
        """Get the list of 1D Laplacian operators."""
        return [getattr(self, f"laplacian_{i}") for i in range(self.ndim)]

    def _compute_inverse_diagonals(self):
        """
        Precompute the inverse diagonals of the Laplacian operator on the Grid mesh. Must be implemented by subclasses.

        For PseudoInverse class, the diagonals are in FFT/SVD spaces, which corresponds to the eigenvalues of the Laplacian operator.
        For IterativeSolver class, this is simply the inverse diagonal of the original 1D Laplacian operators.
        """
        raise NotImplementedError(
            "Subclasses must implement _compute_inverse_diagonals"
        )

    def forward(self, f: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        """
        For PseudoInverseBase: apply the pseudoinverse (with a cutoff) Laplacian operator to the input tensor.
        For IterativeSolverBase: solve the linear system Au = f, where A is the Laplacian operator and f is the right-hand side, q0 is the initial guess.

        Args:
            value: right-hand-side of the linear operator. This is a tensor with `len(operators)` dimensions, where each dimension corresponds to one of the linear operators.
            q0: initial guess for the solution. Not used in PseudoInverseBase, but may be used in IterativeSolverBase.

        Returns:
            A^{*} rhs, where A^{*} is either the pseudoinverse of the Laplacian operator (eigen-expansion with a cut-off) or the iterative solver's A_h^{-1}'s approximation.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def solve(self, f: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement solve method")


class PseudoInverseBase(SolverBase):
    """
    Base class for pseudoinverse of the Laplacian operator on a given Grid.

    This class applies the pseudoinverse of the Laplacian operator using the
    "fast diagonalization method" for separable linear operators.

    The application of a linear operator (the inverse of Laplacian)
    can be written as a sum of operators on each axis.
    Such linear operators are *separable*, and can be written as a sum of tensor
    products, e.g., `operators = [A, B]` corresponds to the linear operator
    A ⊗ I + I ⊗ B, where the tensor product ⊗ indicates a separation between
    operators applied along the first and second axis.

    This function computes matrix-valued functions of such linear operators via
    the "fast diagonalization method" [1]:
    F(A ⊗ I + I ⊗ B)
    = (X(A) ⊗ X(B)) F(Λ(A) ⊗ I + I ⊗ Λ(B)) (X(A)^{-1} ⊗ X(B)^{-1})

    where X(A) denotes the matrix of eigenvectors of A and Λ(A) denotes the
    (diagonal) matrix of eigenvalues. The function `F` is easy to compute in
    this basis, because matrix Λ(A) ⊗ I + I ⊗ Λ(B) is diagonal.

    References:
    [1] Lynch, R. E., Rice, J. R. & Thomas, D. H. Direct solution of partial
        difference equations by tensor product methods. Numer. Math. 6, 185-199
        (1964). https://paperpile.com/app/p/b7fdea4e-b2f7-0ada-b056-a282325c3ecf
    """

    def __init__(
        self,
        grid: grids.Grid,
        bc: Optional[BoundaryConditions] = None,
        dtype: torch.dtype = torch.float32,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-8,
        **kwargs,
    ):
        super().__init__(grid, bc, dtype, laplacians, tol, **kwargs)

    @property
    def eigenvectors(self) -> List[torch.Tensor]:
        """Get the list of eigenvector matrices."""
        if hasattr(self, "eigenvectors_0"):
            return [getattr(self, f"eigenvectors_{i}") for i in range(self.ndim)]
        return []

    def _filter_eigenvalues(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Apply a cutoff function to the eigenvalues.
        """
        return torch.where(torch.abs(eigenvalues) > self.tol, 1 / eigenvalues, 0)

    def solve(self, f: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        return self.forward(f, q0)


class PseudoInverseFFT(PseudoInverseBase):
    """
    PseudoInverse implementation using complex FFT.

    This implementation uses standard FFT for complex-valued computations.
    Requires circulant operators.
    Scales like O(N**d * log(N)) for d N-dimensional operators.
    """

    def __init__(
        self,
        grid: grids.Grid,
        bc: Optional[BoundaryConditions] = None,
        dtype: torch.dtype = torch.float32,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-8,
        **kwargs,
    ):
        super().__init__(grid, bc, dtype, laplacians, tol, **kwargs)

        self.ifft = partial(fft.ifftn, s=grid.shape)
        self.fft = partial(fft.fftn, dim=tuple(range(-grid.ndim, 0)))

    def _compute_inverse_diagonals(self):
        """
        Precompute eigenvalues using FFT for FFT implementation.
        """
        eigenvalues = [fft.fft(op[:, 0]) for op in self.operators]

        summed_eigenvalues = outer_sum(eigenvalues)
        inverse_eigvs = torch.asarray(self._filter_eigenvalues(summed_eigenvalues))

        if inverse_eigvs.shape != summed_eigenvalues.shape:
            raise ValueError(
                "output shape from func() does not match input shape: "
                f"{inverse_eigvs.shape} vs {summed_eigenvalues.shape}"
            )
        self.register_buffer("inverse_diag", inverse_eigvs, persistent=True)

    def forward(self, f: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse in frequency domain and return to real space.
        """

        return self.ifft(self.inverse_diag * self.fft(f)).real


class PseudoInverseRFFT(PseudoInverseFFT):
    """
    PseudoInverse implementation using Real FFT.

    This implementation uses RFFT for faster computation with real-valued data.
    Requires circulant operators and an even-sized last axis.
    Scales like O(N**d * log(N)) for d N-dimensional operators.
    """

    def __init__(
        self,
        grid: grids.Grid,
        bc: Optional[BoundaryConditions] = None,
        dtype: torch.dtype = torch.float32,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-8,
        **kwargs,
    ):
        super().__init__(grid, bc, dtype, laplacians, tol, **kwargs)

        if grid.shape[-1] % 2:
            raise ValueError("RFFT implementation requires even-sized last axis")

        self.ifft = partial(fft.irfftn, s=grid.shape)
        self.fft = partial(fft.rfftn, dim=tuple(range(-grid.ndim, 0)))

    def _compute_inverse_diagonals(self):
        """
        Precompute eigenvalues using FFT for RFFT implementation.
        """
        eigenvalues = [fft.fft(op[:, 0]) for op in self.operators[:-1]] + [
            fft.rfft(self.operators[-1][:, 0])
        ]

        summed_eigenvalues = outer_sum(eigenvalues)
        inverse_eigvs = torch.asarray(self._filter_eigenvalues(summed_eigenvalues))

        if inverse_eigvs.shape != summed_eigenvalues.shape:
            raise ValueError(
                "output shape from func() does not match input shape: "
                f"{inverse_eigvs.shape} vs {summed_eigenvalues.shape}"
            )
        self.register_buffer("inverse_diag", inverse_eigvs, persistent=True)


class PseudoInverseMatmul(PseudoInverseBase):
    """
    PseudoInverse implementation using matrix multiplication in eigenspace.

    This implementation directly diagonalizes dense matrices for each linear operator.
    Requires hermitian operators.
    Scales like O(N**(d+1)) for d N-dimensional operators, but makes good use of matmul hardware.
    """

    def __init__(
        self,
        grid: grids.Grid,
        bc: Optional[BoundaryConditions] = None,
        dtype: torch.dtype = torch.float32,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-8,
        **kwargs,
    ):
        super().__init__(grid, bc, dtype, laplacians, tol, **kwargs)

    def _compute_inverse_diagonals(self):
        """
        Precompute eigenvalues and eigenvectors using matrix diagonalization.
        """
        eigenvalues, eigenvectors = zip(*map(torch.linalg.eigh, self.operators))

        summed_eigenvalues = outer_sum(eigenvalues)
        inverse_eigvs = torch.asarray(self._filter_eigenvalues(summed_eigenvalues))

        if inverse_eigvs.shape != summed_eigenvalues.shape:
            raise ValueError(
                "output shape from func() does not match input shape: "
                f"{inverse_eigvs.shape} vs {summed_eigenvalues.shape}"
            )
        self.register_buffer("inverse_diag", inverse_eigvs, persistent=True)

        # Register eigenvectors
        for i, evecs in enumerate(eigenvectors):
            self.register_buffer(f"eigenvectors_{i}", evecs, persistent=True)

    def forward(self, f: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse in SVD space and return to real space.
        """
        out = f
        # Forward transform: contract along spatial dimensions from the end
        for vectors in self.eigenvectors:
            out = torch.tensordot(out, vectors, dims=([-2], [0]))  # type: ignore
        out *= torch.as_tensor(self.inverse_diag, dtype=out.dtype)
        # Inverse transform: contract along spatial dimensions from the end
        for vectors in self.eigenvectors:
            out = torch.tensordot(out, vectors, dims=([-2], [1]))  # type: ignore

        return out


class PseudoInverse(nn.Module):
    """
    Factory class for creating PseudoInverse solvers with different implementations.

    This class automatically selects the appropriate implementation based on the
    parameters and grid properties, or creates a specific implementation if requested.

    Args:
        grid: Grid object describing the spatial domain.
        bc: Boundary conditions for the Laplacian operator (for pressure).
        dtype: Tensor data type.
        hermitian: whether or not all linear operator are Hermitian (i.e., symmetric in the real valued case).
        circulant: If True, bc is periodical
        implementation: One of ['fft', 'rfft', 'matmul']. If None, automatically selects based on grid properties.
        cutoff: Minimum eigenvalue to invert.
        laplacians: Precomputed Laplacian operators. If None, they are computed from the grid during initialization.

    implementation: how to implement fast diagonalization:
        - 'matmul': scales like O(N**(d+1)) for d N-dimensional operators, but
        makes good use of matmul hardware. Requires hermitian=True.
        - 'fft': scales like O(N**d * log(N)) for d N-dimensional operators.
        Requires circulant=True.
        - 'rfft': use the RFFT instead of the FFT. This is a little faster than
        'fft' but also has slightly larger error. It currently requires an even
        sized last axis and circulant=True.

    Returns:
        An instance of the appropriate PseudoInverse implementation.
    """

    def __new__(
        cls,
        grid: grids.Grid,
        bc: Optional[BoundaryConditions] = None,
        dtype: torch.dtype = torch.float32,
        hermitian: bool = True,
        circulant: bool = True,
        implementation: Optional[str] = None,
        laplacians: Optional[List[torch.Tensor]] = None,
        cutoff: float = 1e-8,
        **kwargs,
    ):
        # Auto-select implementation if not specified
        if implementation is None:
            implementation = "rfft" if circulant else "matmul"

        # if the last axis is odd, we cannot use rfft
        if implementation == "rfft" and grid.shape[-1] % 2:
            implementation = "fft" if circulant else "matmul"

        # Validate implementation requirements
        if implementation in ["rfft", "fft"] and not circulant:
            raise ValueError(
                f"non-circulant operators not yet supported with implementation='{implementation}'"
            )
        if implementation in ["matmul", "svd"] and not hermitian:
            raise ValueError("matmul implementation requires hermitian=True. ")

        # Create the appropriate implementation
        if implementation == "rfft":
            return PseudoInverseRFFT(grid, bc, dtype, laplacians, cutoff, **kwargs)
        elif implementation == "fft":
            return PseudoInverseFFT(grid, bc, dtype, laplacians, cutoff, **kwargs)
        elif implementation == "matmul":
            return PseudoInverseMatmul(grid, bc, dtype, laplacians, cutoff, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported implementation: {implementation}")


class IterativeSolver(SolverBase):
    """
    Base class for iterative solvers that apply a separable Laplacian
    operator to a tensor `u` of shape (..., *grid.shape) and solve
    the linear system Au = f.

    Args:
        grid: Grid object describing the spatial domain.
        bc: Boundary conditions for the Laplacian operator (for pressure).
        dtype: Tensor data type.
        laplacians: Precomputed Laplacian operators. If None, they are computed from
            the grid during initialization.
        tol: Tolerance for the iterative solver's relative residual.
        max_iter: Maximum number of iterations for the iterative solver.
        check_iter: Frequency of checking the residual norm during iterations.
        record_residuals: If True, record the residual norms during iterations.
    """

    def __init__(
        self,
        grid: Grid,
        bc: BoundaryConditions,
        dtype: torch.dtype = torch.float64,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        check_iter: int = 10,
        record_residuals: bool = False,
    ):
        super().__init__(grid, bc, dtype, laplacians, tol)

        self.max_iter = max_iter
        self.stop_iter = max_iter
        self.check_iter = check_iter
        self.record_residuals = record_residuals
        self.residual_norms = [1.0]  # relative residual

    def _compute_inverse_diagonals(self):
        # inverse diagonal of sum of 1D ops
        self.eps = 1e-10  # small value to avoid division by zero
        diag = outer_sum([torch.diag(op) for op in self.operators])
        inv_diag = torch.where(
            torch.abs(diag) > self.eps,
            1.0 / diag + self.eps,
            torch.zeros_like(diag),
        )
        self.register_buffer("inverse_diag", inv_diag)

    @property
    def residual_norms(self):
        return torch.tensor(self._residual_norms)

    @residual_norms.setter
    def residual_norms(self, value):
        if not hasattr(self, "_residual_norms"):
            self._residual_norms = []
        if isinstance(value, (torch.Tensor, float)):
            self._residual_norms.append(value)
        elif isinstance(value, list):
            self._residual_norms.extend(value)
        else:
            raise TypeError("Residual norms must be a tensor or a list of tensors.")

    def expand_as(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Expand tensor in-place for broadcasting with x."""
        if target.ndim > self.ndim:
            return inp[(slice(None),) + (None,) * self.ndim]
        return inp

    def _apply_laplacian(self, u, operators: Optional[List[torch.Tensor]] = None):
        """
        Apply the separable 2D Laplacian: Au = Lx @ u + u @ Ly.T
        """
        ndim = self.grid.ndim
        out = torch.zeros_like(u.data)
        operators = self.operators if operators is None else operators
        data = u.data
        for i, lap in enumerate(operators):
            dim = i - ndim
            _out = torch.tensordot(data, lap, dims=([dim], [-1]))  # type: ignore
            out += _out.transpose(
                dim, -1
            )  # swap the first axis to the correct position
        return out

    def residual(self, f, u):
        return f - self._apply_laplacian(u)

    def forward(self, f, u, *args, **kwargs) -> torch.Tensor:
        """
        Perform a single iteration step of the iterative solver.

        Args:
            f: Right-hand side tensor.
            u: Current solution tensor.

        Returns:
            Updated solution tensor after one iteration step. u <- u + M(b - Au)
        """
        raise NotImplementedError("forward method must be implemented in subclasses.")

    def solve(
        self,
        f: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        iters: Optional[int] = None,
    ) -> torch.Tensor:
        u = torch.zeros_like(f) if u is None else u
        f_norm = torch.linalg.norm(f)
        iters = self.max_iter if iters is None else iters
        for i in range(iters):
            u_new = self.forward(f, u)
            if i % self.check_iter == 0:
                res_norm = torch.linalg.norm(self.residual(f, u_new)) / f_norm
                if self.record_residuals:
                    self.residual_norms = res_norm.item()
                if res_norm < self.tol:
                    self.stop_iter = i + 1
                    break
            u = u_new
        return u


class Jacobi(IterativeSolver):
    def __init__(
        self,
        grid: Grid,
        bc: BoundaryConditions,
        dtype: torch.dtype = torch.float64,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        check_iter: int = 10,
        record_residuals: bool = False,
        pure_neumann: bool = False,
        interior_only: bool = False,
    ):
        super().__init__(
            grid, bc, dtype, laplacians, tol, max_iter, check_iter, record_residuals
        )
        self.pure_neumann = pure_neumann
        self.interior_only = interior_only
        self._set_up_masks()

    def update(
        self, f: torch.Tensor, u: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        unsqueezed = False
        if u.ndim == self.ndim:
            unsqueezed = True
            u = u.unsqueeze(0)
        u_new = u.clone()
        u_neighbors = torch.zeros(
            *u.shape, 2 * self.ndim, dtype=u.dtype, device=u.device
        )
        mask = mask.to(u.device)
        # idx = 0 left idx = 1 right
        # idx = 2 down idx = 3 up
        Lx, Ly = self.operators
        u_neighbors[..., 1:, :, 0] = Lx.diagonal(-1)[:, None] * u[..., :-1, :]
        u_neighbors[..., :-1, :, 1] = Lx.diagonal(1)[:, None] * u[..., 1:, :]
        u_neighbors[..., 1:, 2] = Ly.diagonal(-1)[None, :] * u[..., :, :-1]
        u_neighbors[..., :-1, 3] = Ly.diagonal(1)[None, :] * u[..., :, 1:]
        update = f - u_neighbors.sum(dim=-1)
        Dinv = torch.as_tensor(self.inverse_diag).expand_as(update)
        mask = mask.expand_as(update)
        u_new[mask] = Dinv[mask] * update[mask]
        return u_new.squeeze(0) if unsqueezed else u_new

    def _set_up_masks(self):
        self.masks = []
        mask = torch.zeros(self.grid.shape, dtype=torch.bool)
        if self.interior_only:
            mask[..., 1:-1, 1:-1] = True
        else:
            mask[:] = True
        self.masks.append(mask)

    def forward(self, f: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Jacobi iteration using explicit neighbor contributions"""
        for mask in self.masks:
            u = self.update(f, u, mask)
        if self.pure_neumann:
            u -= torch.mean(
                u, dim=(-2, -1), keepdim=True
            )  # Remove mean for pure Neumann BC
        return u


class GaussSeidel(Jacobi):
    """
    Gauss-Seidel iterative solver for Laplacian systems.

    The implementation uses red-black ordering to update
    the solution with a mask inherited from the Jacobi class.

    Reference: Long Chen's notes on vectorizing finite difference methods in MATLAB
    https://www.math.uci.edu/~chenlong/226/FDMcode.pdf
    """

    def __init__(
        self,
        grid: Grid,
        bc: BoundaryConditions,
        dtype: torch.dtype = torch.float64,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        check_iter: int = 10,
        record_residuals: bool = False,
        pure_neumann: bool = False,
    ):
        super().__init__(
            grid,
            bc,
            dtype,
            laplacians,
            tol,
            max_iter,
            check_iter,
            record_residuals,
            pure_neumann,
        )

    def _set_up_masks(self):
        nx, ny = self.grid.shape
        ix = torch.arange(nx, device=self.grid.device)
        iy = torch.arange(ny, device=self.grid.device)
        ix, iy = torch.meshgrid(ix, iy, indexing="ij")
        red_mask = (ix + iy) % 2 == 0
        black_mask = ~red_mask
        self.masks = [red_mask, black_mask]

    def forward(self, f: torch.Tensor, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Gauss-Seidel iteration using explicit neighbor contributions.
        The method alternates between red and black masks to update the solution.
        """
        return super().forward(f, u, **kwargs)


class ConjugateGradient(IterativeSolver):
    def __init__(
        self,
        grid: Grid,
        bc: BoundaryConditions,
        dtype: torch.dtype = torch.float64,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
        check_iter: int = 10,
        record_residuals: bool = False,
        pure_neumann: bool = False,
        preconditioner: Optional[str | Callable] = None,
    ):
        super().__init__(
            grid, bc, dtype, laplacians, tol, max_iter, check_iter, record_residuals
        )
        self.pure_neumann = pure_neumann
        # setup preconditioner
        if isinstance(preconditioner, str):
            if preconditioner == "jacobi":
                self.preconditioner = Jacobi(
                    grid, bc, laplacians=laplacians, max_iter=1
                )
            elif preconditioner in ["gauss_seidel", "gs"]:
                self.preconditioner = GaussSeidel(
                    grid, bc, laplacians=laplacians, max_iter=1
                )
            self.inv_diag = self.preconditioner.inverse_diag
        elif callable(preconditioner):
            self.preconditioner = preconditioner
        elif preconditioner is None:
            self.preconditioner = Identity()
        else:
            raise NotImplementedError(
                f"Preconditioner {preconditioner} not implemented."
            )

    def apply_preconditioner(self, r):
        """
        Apply the preconditioner to the residual r.
        If the preconditioner is an Identity, it returns r unchanged.
        """
        return self.preconditioner.forward(r, torch.zeros_like(r))

    def forward(self, u, r, p, rsold):
        """One step of preconditioned conjugate gradient iteration."""
        Ap = self._apply_laplacian(p)

        # Use negative indexing to handle both batched and non-batched cases
        spatial_dims = tuple(range(-self.ndim, 0))

        pAp = torch.sum(p * Ap, dim=spatial_dims)
        alpha = rsold / (pAp + self.eps)

        alpha = self.expand_as(alpha, u)

        u += alpha * p
        r -= alpha * Ap

        z = self.apply_preconditioner(r)
        # z = self.inv_diag.expand_as(r) * r

        rznew = torch.sum(r * z, dim=spatial_dims)
        beta = rznew / (rsold + self.eps)

        beta = self.expand_as(beta, u)

        p = z + beta * p

        return u, r, p, rznew

    def solve(self, f, u=None):
        u = torch.zeros_like(f) if u is None else u
        res = self.residual(f, u)
        z = self.apply_preconditioner(res)
        p = z.clone()

        spatial_dims = tuple(range(-self.ndim, 0))
        f_norm = torch.linalg.norm(f)
        rdotz = torch.sum(res * z, dim=spatial_dims)
        for i in range(1, self.max_iter + 1):
            u, res, p, rdotz = self.forward(u, res, p, rdotz)
            if self.pure_neumann:
                u -= torch.mean(u, dim=(-2, -1), keepdim=True)
            if i % self.check_iter == 0:
                residual_norm = torch.linalg.norm(res) / f_norm
                if residual_norm >= self.residual_norms[-1]:
                    self.preconditioner = Identity()
                if self.record_residuals:
                    self.residual_norms = residual_norm.item()
                if residual_norm < self.tol:
                    self.stop_iter = i
                    break
        return u


class MultigridSolver(IterativeSolver):
    """
    Multilevel V-cycle multigrid solver for Neumann Laplacian (pressure projection).

    On MAC grids, the velocity's multigrid needs specially designed prolongation and restriction operators.

    References:
    Long Chen's lecture notes on MAC grids and how to implement the multigrid for the Stokes system:
    https://www.math.uci.edu/~chenlong/226/MACcode.pdf
    """

    def __init__(
        self,
        grid: Grid,
        bc: BoundaryConditions,
        dtype: torch.dtype = torch.float64,
        laplacians: Optional[List[torch.Tensor]] = None,
        tol: float = 1e-6,
        max_iter: int = 10,
        check_iter: int = 1,
        levels: int = 2,
        pre_smooth: int = 1,
        post_smooth: int = 1,
        record_residuals: bool = False,
        pure_neumann: bool = False,
    ):
        super().__init__(
            grid, bc, dtype, laplacians, tol, max_iter, check_iter, record_residuals
        )
        self.levels = levels
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.pure_neumann = pure_neumann
        # build grids hierarchy
        self.grids: List[Grid] = [grid]
        ops = [laplacians or getattr(self, "operators")]

        for _ in range(1, levels):
            # 0 is the finest grid, -1 is the coarsest
            prev = self.grids[-1]
            shape = tuple(s // 2 for s in prev.shape)
            c_grid = Grid(shape, domain=grid.domain, device=grid.device)
            self.grids.append(c_grid)
            ops.append(fdm.set_laplacian_matrix(c_grid, bc, dtype=dtype))
        self._register_level_operators(ops)

        # smoothers per level (Gauss-Seidel)
        self.smoothers = nn.ModuleList(
            [
                GaussSeidel(
                    g,
                    bc,
                    dtype,
                    tol=tol,
                    laplacians=op,
                    max_iter=1,
                )
                for g, op in zip(self.grids, self.ops)
            ]
        )
        # precompute coarse operator and direct solver for coarsest level
        L0, L1 = self.ops[-1]
        I0 = torch.eye(L0.size(0), dtype=dtype)
        I1 = torch.eye(L1.size(0), dtype=dtype)
        A_coarse = torch.kron(I1, L0) + torch.kron(L1, I0)
        self.register_buffer("A_coarse", A_coarse, persistent=True)

    def _register_level_operators(self, operators: List[List[torch.Tensor]]):
        """Register operators for a specific level as buffers."""
        for lvl, ops in enumerate(operators):
            for i, op in enumerate(ops):
                self.register_buffer(f"ops_level_{lvl}_{i}", op, persistent=True)

    def _get_level_operators(self, level: int) -> List[torch.Tensor]:
        """Get operators for a specific level."""
        assert (
            0 <= level < self.levels
        ), f"Invalid level: {level}. Must be in range [0, {self.levels - 1}]."
        operators = []
        for i in range(self.ndim):
            operators.append(getattr(self, f"ops_level_{level}_{i}"))
        return operators

    @property
    def ops(self) -> List[List[torch.Tensor]]:
        """Get all operators for all levels (for backward compatibility)."""
        return [self._get_level_operators(lvl) for lvl in range(self.levels)]

    def restrict(self, r):
        # full-weighting restriction
        return 0.25 * (
            r[..., ::2, ::2]
            + r[..., 1::2, ::2]
            + r[..., ::2, 1::2]
            + r[..., 1::2, 1::2]
        )

    def prolong(self, e):
        """
        Prolongation (interpolation) operator - transpose of restriction.
        For full-weighting restriction, the adjoint prolongation distributes
        each coarse grid value to the 4 corresponding fine grid points.
        """
        *batch, nx_c, ny_c = e.shape
        nx_f, ny_f = nx_c * 2, ny_c * 2
        up = torch.zeros(*batch, nx_f, ny_f, device=e.device, dtype=e.dtype)

        # Distribute each coarse grid point to the corresponding 2x2 fine grid region
        # This is the adjoint of full-weighting restriction
        up[..., ::2, ::2] += e  # top-left
        up[..., 1::2, ::2] += e  # top-right
        up[..., ::2, 1::2] += e  # bottom-left
        up[..., 1::2, 1::2] += e  # bottom-right

        return up

    def _coarse_solve(self, r):
        """
        Direct solve on coarsest grid using precomputed dense operator.
        """
        # flatten residual: (batch_size, nx, ny) -> (batch_size, nx*ny)
        batch_size, nx, ny = r.shape
        r_vec = r.reshape(batch_size, -1)  # (batch_size, nx*ny)
        r_vec = r_vec.T  # (nx*ny, batch_size)

        # solve A @ x = r for each sample in the batch
        x_vec = torch.linalg.solve(self.A_coarse, r_vec)  # (nx*ny, batch_size)

        x_vec = x_vec.T  # (batch_size, nx*ny)
        return x_vec.reshape(batch_size, nx, ny)

    def v_cycle(self, level, f, u):
        all_ops = self.ops
        # pre-smoothing
        for _ in range(self.pre_smooth):
            u = self.smoothers[level].forward(f, u)
        # residual
        if level == 0:
            r = f - self._apply_laplacian(u)
        else:
            r = f - self._apply_laplacian(u, operators=all_ops[level])
        if level == self.levels - 1:
            # coarsest: direct solve via dense matrix
            e = self._coarse_solve(r)
        else:
            # restrict
            rc = self.restrict(r)
            ec0 = torch.zeros_like(rc)
            ec = self.v_cycle(level + 1, rc, ec0)
            e = self.prolong(ec)
        # correction
        u += e
        if self.pure_neumann:
            # ensure Neumann BC by removing mean
            u -= torch.mean(u, dim=(-2, -1), keepdim=True)
        # post-smoothing
        for _ in range(self.post_smooth):
            u = self.smoothers[level].forward(f, u)
        return u

    def forward(self, f, u=None):
        u = torch.zeros_like(f) if u is None else u
        return self.v_cycle(0, f, u)
