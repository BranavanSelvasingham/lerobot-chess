#!/usr/bin/env python

# Copyright 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class SE3:
    """Simple SE(3) rigid transform wrapper using 4x4 homogeneous matrices."""

    T: np.ndarray  # shape (4, 4)

    @staticmethod
    def from_rt(R: np.ndarray, t: np.ndarray) -> """Create from rotation (3x3) and translation (3,).""":
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)
        return SE3(T)

    @staticmethod
    def identity() -> """Return identity transform.""":
        return SE3(np.eye(4, dtype=float))

    def as_rt(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.T[:3, :3].copy(), self.T[:3, 3].copy()

    def inverse(self) -> """Return inverse transform.""":
        R, t = self.as_rt()
        R_inv = R.T
        t_inv = -R_inv @ t
        return SE3.from_rt(R_inv, t_inv)

    def __matmul__(self, other: "SE3 | np.ndarray") -> "SE3 | np.ndarray":
        if isinstance(other, SE3):
            return SE3(self.T @ other.T)
        return self.T @ other


def rodrigues(rotvec: Iterable[float]) -> np.ndarray:
    """Convert axis-angle (rotation vector) to 3x3 rotation matrix.

    Args:
        rotvec: (3,) rotation vector in radians.
    """
    rv = np.asarray(rotvec, dtype=float).reshape(3)
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=float
    )
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def homogenize(points_xy: np.ndarray) -> np.ndarray:
    """Convert (N,2) to homogeneous (3,N)."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be of shape (N,2)")
    return np.vstack([pts.T, np.ones((1, pts.shape[0]))])


def dehomogenize(points_h: np.ndarray) -> np.ndarray:
    """Convert homogeneous (3,N) to (N,2)."""
    ph = np.asarray(points_h, dtype=float)
    if ph.ndim != 2 or ph.shape[0] not in (3, 4):
        raise ValueError("points_h must be of shape (3,N) or (4,N)")
    ph = ph[:3, :]
    w = ph[2:3, :]
    w = np.where(np.abs(w) < 1e-12, 1.0, w)
    xy = ph[:2, :] / w
    return xy.T


def estimate_homography(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """DLT homography from 4+ point correspondences.

    Args:
        src_xy: (N,2) source points
        dst_xy: (N,2) destination points
    Returns:
        H: (3,3) homography such that x' ~ H x
    """
    src = np.asarray(src_xy, dtype=float)
    dst = np.asarray(dst_xy, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 4:
        raise ValueError("Need at least 4 point pairs of shape (N,2)")
    N = src.shape[0]
    A = []
    for i in range(N):
        x, y = src[i]
        X, Y = dst[i]
        A.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
    A = np.asarray(A, dtype=float)
    # Solve Ah=0 using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    # Normalize
    if np.abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Apply homography H (3x3) to (N,2) points."""
    return dehomogenize(H @ homogenize(pts_xy))


def compose(*transforms: SE3) -> SE3:
    """Compose multiple SE3 transforms left-to-right."""
    T = SE3.identity()
    for t in transforms:
        T = T @ t
    return T


