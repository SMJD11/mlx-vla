"""Pure-NumPy trajectory tokenizers and course-heading unicycle action space.

Matches NVIDIA Alpamayo2-Edge's PyTorch float64/float32 implementations
(DeltaTrajectoryTokenizer, UnicycleAccelCurvatureActionSpace, and
UnicycleAccelCurvatureCourseActionSpace) without requiring PyTorch.
"""

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

_LOW_SPEED_CURVATURE_THRESHOLD_MPS = 0.6
_COURSE_LOW_SPEED_CURVATURE_THRESHOLD_MPS = 0.1
_COURSE_FADE_START_MPS = 0.5
_COURSE_FADE_END_MPS = 1.0
_MAX_COURSE_OFFSET_RAD = 0.05


def so3_to_yaw_np(rot_mat: np.ndarray) -> np.ndarray:
    return np.arctan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])


def round_2pi_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def unwrap_angle_np(phi: np.ndarray) -> np.ndarray:
    d = round_2pi_np(np.diff(phi, axis=-1))
    return np.concatenate([phi[..., :1], phi[..., :1] + np.cumsum(d, axis=-1)], axis=-1)


def rot_2d_to_3d_np(angle: np.ndarray) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    z = np.zeros_like(c)
    o = np.ones_like(c)
    row0 = np.stack([c, -s, z], axis=-1)
    row1 = np.stack([s, c, z], axis=-1)
    row2 = np.stack([z, z, o], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def _first_order_D(n: int, dtype) -> np.ndarray:
    d = np.zeros((max(n - 1, 0), n), dtype=dtype)
    rows = np.arange(max(n - 1, 0))
    d[rows, rows] = -1.0
    d[rows, rows + 1] = 1.0
    return d


def _second_order_D(n: int, dtype) -> np.ndarray:
    d = np.zeros((max(n - 2, 0), n), dtype=dtype)
    rows = np.arange(max(n - 2, 0))
    d[rows, rows] = -1.0
    d[rows, rows + 1] = 2.0
    d[rows, rows + 2] = -1.0
    return d


def _third_order_D(n: int, dtype) -> np.ndarray:
    d = np.zeros((max(n - 3, 0), n), dtype=dtype)
    rows = np.arange(max(n - 3, 0))
    d[rows, rows] = -1.0
    d[rows, rows + 1] = 3.0
    d[rows, rows + 2] = -3.0
    d[rows, rows + 3] = 1.0
    return d


def _construct_DTD(
    n: int,
    lead: Tuple[int, ...],
    dtype,
    w_smooth1: Optional[float] = None,
    w_smooth2: Optional[float] = None,
    w_smooth3: Optional[float] = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> np.ndarray:
    dtd = np.zeros((n, n), dtype=dtype)
    if w_smooth1 is not None:
        d1 = _first_order_D(n, dtype)
        dtd = dtd + np.array(lam / (dt**2) * w_smooth1, dtype=dtype) * (d1.T @ d1)
    if w_smooth2 is not None:
        d2 = _second_order_D(n, dtype)
        dtd = dtd + np.array(lam / (dt**4) * w_smooth2, dtype=dtype) * (d2.T @ d2)
    if w_smooth3 is not None:
        d3 = _third_order_D(n, dtype)
        dtd = dtd + np.array(lam / (dt**6) * w_smooth3, dtype=dtype) * (d3.T @ d3)
    if lead:
        dtd = np.broadcast_to(dtd, (*lead, n, n)).copy()
    return dtd


def _cholesky_solve(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    # Solve (..., N, N) @ x = (..., N) via Cholesky matching torch.cholesky_solve
    L = np.linalg.cholesky(lhs)
    # L @ y = rhs, L^T @ x = y -> np.linalg.solve(lhs, rhs[..., None]) uses LU; solve via L & L^T
    y = np.linalg.solve(L, rhs[..., None])
    x = np.linalg.solve(np.swapaxes(L, -1, -2), y)
    return x[..., 0]


def _solve_single_constraint(
    x_init: np.ndarray,
    x_target: np.ndarray,
    w_smooth1: Optional[float] = None,
    w_smooth2: Optional[float] = None,
    w_smooth3: Optional[float] = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> np.ndarray:
    dtype = x_target.dtype
    *lead, n = x_target.shape
    x_init = np.asarray(x_init, dtype=dtype)
    eye = np.eye(n, dtype=dtype)
    ata = np.broadcast_to(eye, (*lead, n, n)).copy()
    rhs = x_target.copy()
    dtd = _construct_DTD(
        n + 1,
        tuple(lead),
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs = rhs - dtd[..., 1:, 0] * x_init[..., None]
    lhs = ata + dtd[..., 1:, 1:] + np.array(ridge, dtype=dtype) * eye
    sol = _cholesky_solve(lhs, rhs)
    return np.concatenate([x_init[..., None], sol], axis=-1)


def _solve_xs_eq_y(
    s: np.ndarray,
    y: np.ndarray,
    w_smooth1: Optional[float] = None,
    w_smooth2: Optional[float] = None,
    w_smooth3: Optional[float] = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> np.ndarray:
    dtype = y.dtype
    *lead, n = y.shape
    ata = np.zeros((*lead, n, n), dtype=dtype)
    idx = np.arange(n)
    ata[..., idx, idx] = s * s
    rhs = s * y
    dtd = _construct_DTD(
        n,
        tuple(lead),
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    eye = np.eye(n, dtype=dtype)
    cur_ridge = ridge
    while True:
        try:
            lhs = ata + dtd + np.array(cur_ridge, dtype=dtype) * eye
            return _cholesky_solve(lhs, rhs)
        except np.linalg.LinAlgError:
            cur_ridge = max(cur_ridge * 10.0, 1e-8)


def _dxy_theta_to_v_without_v0(
    dxy: np.ndarray,
    theta: np.ndarray,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> np.ndarray:
    dtype = dxy.dtype
    *lead, n, _ = dxy.shape
    g = (2.0 / dt) * dxy
    b_data = g.reshape(*lead, 2 * n)
    a_data = np.zeros((*lead, 2 * n, n + 1), dtype=dtype)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_rows = 2 * np.arange(n)
    sin_rows = 2 * np.arange(n) + 1
    cols = np.arange(n)
    a_data[..., cos_rows, cols] = cos_t[..., :-1]
    a_data[..., cos_rows, cols + 1] = cos_t[..., 1:]
    a_data[..., sin_rows, cols] = sin_t[..., :-1]
    a_data[..., sin_rows, cols + 1] = sin_t[..., 1:]
    ata = np.einsum("...ij,...ik->...jk", a_data, a_data)
    rhs = np.einsum("...ij,...i->...j", a_data, b_data)
    dtd = _construct_DTD(
        n + 1, tuple(lead), dtype=dtype, w_smooth3=1.0, lam=v_lambda, dt=dt
    )
    lhs = ata + dtd + np.array(v_ridge, dtype=dtype) * np.eye(n + 1, dtype=dtype)
    return _cholesky_solve(lhs, rhs)


def _dxy_theta_to_v(
    dxy: np.ndarray,
    theta: np.ndarray,
    v0: np.ndarray,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> np.ndarray:
    dtype = dxy.dtype
    *lead, n, _ = dxy.shape
    g = (2.0 / dt) * dxy
    b_data = g.reshape(*lead, 2 * n)
    a_data = np.zeros((*lead, 2 * n, n + 1), dtype=dtype)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_rows = 2 * np.arange(n)
    sin_rows = 2 * np.arange(n) + 1
    cols = np.arange(n)
    a_data[..., cos_rows, cols] = cos_t[..., :-1]
    a_data[..., cos_rows, cols + 1] = cos_t[..., 1:]
    a_data[..., sin_rows, cols] = sin_t[..., :-1]
    a_data[..., sin_rows, cols + 1] = sin_t[..., 1:]
    ata = np.einsum("...ij,...ik->...jk", a_data, a_data)
    rhs = np.einsum("...ij,...i->...j", a_data[..., :, 1:], b_data)
    rhs = rhs - ata[..., 1:, 0] * v0[..., None]
    dtd = _construct_DTD(
        n + 1, tuple(lead), dtype=dtype, w_smooth3=1.0, lam=v_lambda, dt=dt
    )
    rhs = rhs - dtd[..., 1:, 0] * v0[..., None]
    lhs = (
        ata[..., 1:, 1:]
        + dtd[..., 1:, 1:]
        + np.array(v_ridge, dtype=dtype) * np.eye(n, dtype=dtype)
    )
    sol = _cholesky_solve(lhs, rhs)
    return np.concatenate([v0[..., None], sol], axis=-1)


def _course_offset(chord_xy: np.ndarray, yaw: np.ndarray, chord_dt: float) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    along = cos_yaw * chord_xy[..., 0] + sin_yaw * chord_xy[..., 1]
    across = cos_yaw * chord_xy[..., 1] - sin_yaw * chord_xy[..., 0]
    # Note: torch.sign(0.0) == 0.0 and np.sign(0.0) == 0.0
    offset = np.arctan2(across * np.sign(along), np.abs(along))
    offset = np.where(np.abs(offset) > _MAX_COURSE_OFFSET_RAD, 0.0, offset).astype(
        chord_xy.dtype
    )
    speed = np.linalg.norm(chord_xy, axis=-1) / chord_dt
    fade = np.clip(
        (speed - _COURSE_FADE_START_MPS)
        / (_COURSE_FADE_END_MPS - _COURSE_FADE_START_MPS),
        0.0,
        1.0,
    ).astype(chord_xy.dtype)
    return offset * fade


class UnicycleAccelCurvatureActionSpace:
    """Pure-NumPy UnicycleAccelCurvatureActionSpace."""

    low_speed_curvature_threshold_mps: float = _LOW_SPEED_CURVATURE_THRESHOLD_MPS

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: Tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: Tuple[float, float] = (-0.33, 0.33),
        dt: float = 0.1,
        n_waypoints: int = 64,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
        **_ignored: Any,
    ):
        # Match PyTorch float32 buffer precision on normalization constants
        self.accel_mean = float(np.float32(accel_mean))
        self.accel_std = float(np.float32(accel_std))
        self.curvature_mean = float(np.float32(curvature_mean))
        self.curvature_std = float(np.float32(curvature_std))
        self.accel_bounds = tuple(accel_bounds)
        self.curvature_bounds = tuple(curvature_bounds)
        self.dt = float(dt)
        self.n_waypoints = int(n_waypoints)
        self.theta_lambda = float(theta_lambda)
        self.theta_ridge = float(theta_ridge)
        self.v_lambda = float(v_lambda)
        self.v_ridge = float(v_ridge)
        self.a_lambda = float(a_lambda)
        self.a_ridge = float(a_ridge)
        self.kappa_lambda = float(kappa_lambda)
        self.kappa_ridge = float(kappa_ridge)

    def get_action_space_dims(self) -> Tuple[int, int]:
        return (self.n_waypoints, 2)

    def estimate_t0_states(
        self, traj_history_xyz: np.ndarray, traj_history_rot: np.ndarray
    ) -> Dict[str, np.ndarray]:
        hist_xyz = np.asarray(traj_history_xyz)
        hist_rot = np.asarray(traj_history_rot)
        dxy = hist_xyz[..., 1:, :2] - hist_xyz[..., :-1, :2]
        theta = unwrap_angle_np(so3_to_yaw_np(hist_rot))
        v = _dxy_theta_to_v_without_v0(
            dxy=dxy,
            theta=theta,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        return {"v": v[..., -1]}

    def _initial_heading(self, t0_states: Dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros_like(t0_states["v"])

    def _heading_targets(
        self,
        traj_history_xyz: np.ndarray,
        traj_history_rot: np.ndarray,
        traj_future_xyz: np.ndarray,
        traj_future_rot: np.ndarray,
    ) -> np.ndarray:
        return unwrap_angle_np(so3_to_yaw_np(traj_future_rot))

    def _output_heading(
        self,
        theta: np.ndarray,
        velocity: np.ndarray,
        t0_states: Dict[str, np.ndarray],
    ) -> np.ndarray:
        return theta

    def traj_to_action(
        self,
        traj_history_xyz: np.ndarray,
        traj_history_rot: np.ndarray,
        traj_future_xyz: np.ndarray,
        traj_future_rot: np.ndarray,
        t0_states: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        hist_xyz = np.asarray(traj_history_xyz)
        hist_rot = np.asarray(traj_history_rot)
        fut_xyz = np.asarray(traj_future_xyz)
        fut_rot = np.asarray(traj_future_rot)
        dtype = fut_xyz.dtype

        if t0_states is None:
            t0_states = self.estimate_t0_states(hist_xyz, hist_rot)

        full_xy = np.concatenate([hist_xyz[..., -1:, :2], fut_xyz[..., :2]], axis=-2)
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = _solve_single_constraint(
            x_init=self._initial_heading(t0_states),
            x_target=self._heading_targets(hist_xyz, hist_rot, fut_xyz, fut_rot),
            w_smooth3=1.0,
            lam=self.theta_lambda,
            ridge=self.theta_ridge,
            dt=self.dt,
        )
        v = _dxy_theta_to_v(
            dxy=dxy,
            theta=theta,
            v0=t0_states["v"],
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        dv = (v[..., 1:] - v[..., :-1]) / self.dt
        accel = _solve_xs_eq_y(
            s=np.ones_like(dv),
            y=dv,
            w_smooth2=1.0,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            dt=self.dt,
        )
        dtheta = theta[..., 1:] - theta[..., :-1]
        s_arc = self.dt * v[..., :-1] + 0.5 * (self.dt**2) * accel
        kappa = _solve_xs_eq_y(
            s=s_arc,
            y=dtheta,
            w_smooth2=1.0,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )
        low_speed = (np.abs(v[..., :-1]) < self.low_speed_curvature_threshold_mps) | (
            np.abs(v[..., 1:]) < self.low_speed_curvature_threshold_mps
        )
        kappa = np.clip(kappa, self.curvature_bounds[0], self.curvature_bounds[1])
        kappa = np.where(low_speed, 0.0, kappa).astype(dtype)

        accel_norm = (accel - np.array(self.accel_mean, dtype=dtype)) / np.array(
            self.accel_std, dtype=dtype
        )
        kappa_norm = (kappa - np.array(self.curvature_mean, dtype=dtype)) / np.array(
            self.curvature_std, dtype=dtype
        )
        return np.stack([accel_norm, kappa_norm], axis=-1)

    def action_to_traj(
        self,
        action: np.ndarray,
        traj_history_xyz: np.ndarray,
        traj_history_rot: np.ndarray,
        t0_states: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        act = np.asarray(action)
        hist_xyz = np.asarray(traj_history_xyz)
        hist_rot = np.asarray(traj_history_rot)
        dtype = act.dtype

        accel = act[..., 0] * np.array(self.accel_std, dtype=dtype) + np.array(
            self.accel_mean, dtype=dtype
        )
        kappa = act[..., 1] * np.array(self.curvature_std, dtype=dtype) + np.array(
            self.curvature_mean, dtype=dtype
        )

        if t0_states is None:
            t0_states = self.estimate_t0_states(hist_xyz, hist_rot)

        v0 = t0_states["v"].astype(dtype)
        dt = self.dt
        dt2_half = 0.5 * dt * dt
        half_dt = 0.5 * dt

        velocity = np.concatenate(
            [v0[..., None], v0[..., None] + np.cumsum(accel * dt, axis=-1)],
            axis=-1,
        )
        init_yaw = self._initial_heading(t0_states).astype(dtype)
        theta = np.concatenate(
            [
                init_yaw[..., None],
                init_yaw[..., None]
                + np.cumsum(kappa * velocity[..., :-1] * dt, axis=-1)
                + np.cumsum(kappa * accel * dt2_half, axis=-1),
            ],
            axis=-1,
        )
        x = np.cumsum(velocity[..., :-1] * np.cos(theta[..., :-1]) * half_dt, axis=-1) + np.cumsum(
            velocity[..., 1:] * np.cos(theta[..., 1:]) * half_dt, axis=-1
        )
        y = np.cumsum(velocity[..., :-1] * np.sin(theta[..., :-1]) * half_dt, axis=-1) + np.cumsum(
            velocity[..., 1:] * np.sin(theta[..., 1:]) * half_dt, axis=-1
        )

        batch_dim = hist_xyz.shape[:-2]
        traj_future_xyz = np.zeros((*batch_dim, self.n_waypoints, 3), dtype=hist_xyz.dtype)
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        traj_future_xyz[..., 2] = hist_xyz[..., -1:, 2]

        out_heading = self._output_heading(theta, velocity, t0_states)
        traj_future_rot = rot_2d_to_3d_np(out_heading[..., 1:]).astype(hist_rot.dtype)
        return traj_future_xyz, traj_future_rot


class UnicycleAccelCurvatureCourseActionSpace(UnicycleAccelCurvatureActionSpace):
    """Course-heading unicycle action space with rear-axle sideslip compensation."""

    low_speed_curvature_threshold_mps: float = _COURSE_LOW_SPEED_CURVATURE_THRESHOLD_MPS

    def __init__(self, *args: Any, slip_gradient: float = 2.5e-3, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.slip_gradient = float(slip_gradient)

    def estimate_t0_states(
        self, traj_history_xyz: np.ndarray, traj_history_rot: np.ndarray
    ) -> Dict[str, np.ndarray]:
        hist_xyz = np.asarray(traj_history_xyz)
        hist_rot = np.asarray(traj_history_rot)
        if hist_xyz.shape[-2] < 2:
            raise ValueError(
                f"Course-heading action space needs at least 2 history poses, got {hist_xyz.shape[-2]}."
            )
        t0_states = super().estimate_t0_states(hist_xyz, hist_rot)
        last_yaw = unwrap_angle_np(so3_to_yaw_np(hist_rot[..., -2:, :, :]))
        theta0 = _course_offset(
            chord_xy=hist_xyz[..., -1, :2] - hist_xyz[..., -2, :2],
            yaw=last_yaw.mean(axis=-1),
            chord_dt=self.dt,
        )
        return {**t0_states, "theta": theta0}

    def _initial_heading(self, t0_states: Dict[str, np.ndarray]) -> np.ndarray:
        if "theta" not in t0_states:
            raise ValueError("Course-heading action space requires t0_states['theta'].")
        return t0_states["theta"]

    def _output_heading(
        self,
        theta: np.ndarray,
        velocity: np.ndarray,
        t0_states: Dict[str, np.ndarray],
    ) -> np.ndarray:
        dtheta_dt = np.gradient(theta, self.dt, axis=-1)
        lateral_accel = velocity * dtheta_dt
        slip_change = self.slip_gradient * (lateral_accel - lateral_accel[..., :1])
        return theta - self._initial_heading(t0_states)[..., None] + slip_change

    def _heading_targets(
        self,
        traj_history_xyz: np.ndarray,
        traj_history_rot: np.ndarray,
        traj_future_xyz: np.ndarray,
        traj_future_rot: np.ndarray,
    ) -> np.ndarray:
        xy = np.concatenate(
            [traj_history_xyz[..., -1:, :2], traj_future_xyz[..., :2]], axis=-2
        )
        rot = np.concatenate(
            [traj_history_rot[..., -1:, :, :], traj_future_rot], axis=-3
        )
        yaw = unwrap_angle_np(so3_to_yaw_np(rot))
        interior_offset = _course_offset(
            chord_xy=xy[..., 2:, :] - xy[..., :-2, :],
            yaw=yaw[..., 1:-1],
            chord_dt=2.0 * self.dt,
        )
        last_offset = _course_offset(
            chord_xy=xy[..., -1, :] - xy[..., -2, :],
            yaw=yaw[..., -2:].mean(axis=-1),
            chord_dt=self.dt,
        )
        return yaw[..., 1:] + np.concatenate(
            [interior_offset, last_offset[..., None]], axis=-1
        )


def build_action_space(cfg: Mapping[str, Any]) -> UnicycleAccelCurvatureActionSpace:
    target = str(cfg.get("_target_", ""))
    if "Course" in target:
        return UnicycleAccelCurvatureCourseActionSpace(**cfg)
    return UnicycleAccelCurvatureActionSpace(**cfg)


class DeltaTrajectoryTokenizer:
    """Pure-NumPy port of Alpamayo2-Edge DeltaTrajectoryTokenizer."""

    def __init__(
        self,
        ego_xyz_min: Tuple[float, float, float] = (-4.0, -4.0, -10.0),
        ego_xyz_max: Tuple[float, float, float] = (4.0, 4.0, 10.0),
        ego_yaw_min: float = -math.pi,
        ego_yaw_max: float = math.pi,
        num_bins: int = 1000,
        predict_yaw: bool = False,
        pad_origin_at_beginning: bool = False,
        **_ignored: Any,
    ):
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.ego_yaw_min = ego_yaw_min
        self.ego_yaw_max = ego_yaw_max
        self.num_bins = num_bins
        self._predict_yaw = predict_yaw
        self.pad_origin_at_beginning = pad_origin_at_beginning

    def encode(
        self,
        fut_xyz: np.ndarray,
        fut_rot: Optional[np.ndarray] = None,
        **_ignored: Any,
    ) -> np.ndarray:
        xyz = np.asarray(fut_xyz)
        if self.pad_origin_at_beginning:
            xyz = np.pad(xyz, ((0, 0), (1, 0), (0, 0)), mode="constant")
        dxyz = xyz[:, 1:] - xyz[:, :-1]
        xyz_min = np.asarray(self.ego_xyz_min, dtype=dxyz.dtype)
        xyz_max = np.asarray(self.ego_xyz_max, dtype=dxyz.dtype)
        scaled = (dxyz - xyz_min) / (xyz_max - xyz_min)
        bins = np.clip(
            np.rint(scaled * (self.num_bins - 1)).astype(np.int64),
            0,
            self.num_bins - 1,
        )
        if not self._predict_yaw:
            return bins.reshape(bins.shape[0], -1)

        rot = np.asarray(fut_rot)
        yaw = np.arctan2(rot[..., 0, 1], rot[..., 0, 0])
        if self.pad_origin_at_beginning:
            yaw = np.pad(yaw, ((0, 0), (1, 0)), mode="constant")
        dyaw = np.arctan2(
            np.sin(yaw[:, 1:] - yaw[:, :-1]), np.cos(yaw[:, 1:] - yaw[:, :-1])
        )
        yaw_scaled = (dyaw - self.ego_yaw_min) / (self.ego_yaw_max - self.ego_yaw_min)
        yaw_bins = np.clip(
            np.rint(yaw_scaled * (self.num_bins - 1)).astype(np.int64),
            0,
            self.num_bins - 1,
        )
        xyzw = np.concatenate([bins, yaw_bins[..., None]], axis=-1)
        return xyzw.reshape(xyzw.shape[0], -1)


def fuse_history_traj_tokens(
    input_ids_np: np.ndarray,
    ego_history_xyz: np.ndarray,
    ego_history_rot: np.ndarray,
    history_tokenizer: DeltaTrajectoryTokenizer,
    traj_ids: Mapping[str, int],
) -> np.ndarray:
    """Replace <|traj_history|> placeholders with encoded history trajectory token IDs."""
    ids = np.array(input_ids_np, dtype=np.int64, copy=True)
    xyz = np.asarray(ego_history_xyz)
    rot = np.asarray(ego_history_rot)
    if xyz.ndim == 3:
        xyz = xyz[:, None, :, :]
        rot = rot[:, None, :, :, :]
    batch_size, n_traj = xyz.shape[:2]
    flat_xyz = xyz.reshape(batch_size * n_traj, *xyz.shape[2:])
    flat_rot = rot.reshape(batch_size * n_traj, *rot.shape[2:])
    encoded = history_tokenizer.encode(fut_xyz=flat_xyz, fut_rot=flat_rot) + int(
        traj_ids["history_id0"]
    )
    encoded = encoded.reshape(batch_size, -1)

    pad_id = int(traj_ids["history_pad"])
    for b in range(batch_size):
        mask = ids[b] == pad_id
        count = int(mask.sum())
        if count != encoded.shape[1]:
            raise ValueError(
                f"Row {b} has {count} <|traj_history|> placeholders, expected {encoded.shape[1]}"
            )
        ids[b, mask] = encoded[b]
    return ids
