"""
masked_gradient.py

A drop-in replacement for np.gradient() on 2D arrays that excludes zero
values (treated as "missing"/invalid data) from the finite-difference
computation.

Why: np.gradient() computes central differences using immediate
neighbors regardless of whether those neighbors are meaningful. If zeros
represent missing/invalid data, they get treated as real values and
produce large, artificial jumps in the gradient right at the boundary
between real data and zero-filled regions.

This function instead:
  - Builds a validity mask (nonzero by default, or pass your own mask).
  - Uses a central difference when BOTH neighbors along an axis are valid.
  - Falls back to a one-sided (forward/backward) difference when only
    ONE neighbor is valid.
  - Returns NaN when the point itself is invalid, or when neither
    neighbor is valid.
"""

import numpy as np


def masked_gradient(f, mask=None, spacing=(1.0, 1.0)):
    """
    Compute the gradient of a 2D array while excluding zero (or masked)
    values from the finite-difference stencil.

    Parameters
    ----------
    f : array_like, shape (m, n)
        Input 2D array.
    mask : array_like of bool, shape (m, n), optional
        True where data is valid. Defaults to `f != 0`.
    spacing : tuple of float, optional
        Grid spacing along (axis0, axis1). Defaults to (1.0, 1.0).

    Returns
    -------
    grad_axis0, grad_axis1 : ndarray, shape (m, n)
        Gradients along axis 0 (rows) and axis 1 (columns), matching the
        return convention of np.gradient. Entries are NaN wherever the
        gradient could not be computed from valid neighbors.
    """
    f = np.asarray(f, dtype=float)
    if mask is None:
        mask = f != 0
    mask = np.asarray(mask, dtype=bool)

    grad0 = _grad_axis(f, mask, axis=0, spacing=spacing[0])
    grad1 = _grad_axis(f, mask, axis=1, spacing=spacing[1])
    return grad0, grad1


def _slice_along(ndim, axis, start=None, stop=None):
    idx = [slice(None)] * ndim
    idx[axis] = slice(start, stop)
    return tuple(idx)


def _grad_axis(data, valid, axis, spacing):
    n = data.shape[axis]
    grad = np.full(data.shape, np.nan)

    if n < 2:
        return grad  # nothing to differentiate along this axis

    # ---- interior points: indices 1 .. n-2 ----
    center = _slice_along(data.ndim, axis, 1, n - 1)
    left = _slice_along(data.ndim, axis, 0, n - 2)
    right = _slice_along(data.ndim, axis, 2, n)

    v_left = valid[left]
    v_right = valid[right]

    both = v_left & v_right
    only_right = v_right & ~v_left
    only_left = v_left & ~v_right

    g_center = np.full(data[center].shape, np.nan)
    g_center[both] = (data[right][both] - data[left][both]) / (2.0 * spacing)
    g_center[only_right] = (data[right][only_right] - data[center][only_right]) / spacing
    g_center[only_left] = (data[center][only_left] - data[left][only_left]) / spacing
    grad[center] = g_center

    # ---- left edge (index 0): forward diff using points 0 and 1 ----
    idx0 = _slice_along(data.ndim, axis, 0, 1)
    idx1 = _slice_along(data.ndim, axis, 1, 2)
    ok = valid[idx0] & valid[idx1]
    g0 = np.full(data[idx0].shape, np.nan)
    g0[ok] = (data[idx1][ok] - data[idx0][ok]) / spacing
    grad[idx0] = g0

    # ---- right edge (index n-1): backward diff using points n-2, n-1 ----
    idxm1 = _slice_along(data.ndim, axis, n - 1, n)
    idxm2 = _slice_along(data.ndim, axis, n - 2, n - 1)
    ok = valid[idxm1] & valid[idxm2]
    gm = np.full(data[idxm1].shape, np.nan)
    gm[ok] = (data[idxm1][ok] - data[idxm2][ok]) / spacing
    grad[idxm1] = gm

    # A point with no valid data of its own has no meaningful gradient
    grad[~valid] = np.nan

    return grad


if __name__ == "__main__":
    # --- demo ---
    np.set_printoptions(precision=2, suppress=True)

    f = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.1, 2.1, 0.0, 4.1, 5.1],   # a zero "hole" in the middle
        [1.2, 2.2, 3.2, 4.2, 5.2],
        [0.0, 0.0, 0.0, 4.3, 5.3],   # a zero-filled region (e.g. missing data)
        [1.4, 2.4, 3.4, 4.4, 5.4],
    ])

    print("Input array:")
    print(f)

    gy_naive, gx_naive = np.gradient(f)
    print("\nnp.gradient axis-1 (naive, zeros treated as real data):")
    print(gx_naive)

    gy, gx = masked_gradient(f)
    print("\nmasked_gradient axis-1 (zeros excluded):")
    print(gx)
